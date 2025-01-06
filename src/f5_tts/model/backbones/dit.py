"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
    Attention,
    CrossAttnProcessor,
    createAudioTextMask
)
from f5_tts.model.chunk_attn import ChunkLlamaRotaryEmbedding

# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim, padding_idx = 0)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        seq_len = text.size(1)
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        text_mask = (text != 0).float()

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            if text_mask is not None:
                text = text.masked_fill(text_mask.unsqueeze(-1) == 0, 0.0)

            # convnextv2 blocks
            for block in self.text_blocks:
                text = block(text, text_mask)
                if text_mask is not None:
                    text = text.masked_fill(text_mask.unsqueeze(-1) == 0, 0.0)

        return text, text_mask


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, heads, dim_head):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2, out_dim)
        self.text_proj = nn.Linear(text_dim, out_dim)
        self.cross_attention = Attention(
            processor=CrossAttnProcessor(),
            dim=out_dim,
            heads=heads,
            dim_head=dim_head
        )
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, 
                x: float["b n d"], 
                cond: float["b n d"], 
                text_embed: float["b n d"], 
                drop_audio_cond=False,
                audio_mask = None,
                text_mask = None):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond), dim=-1))
        text_embed = self.text_proj(text_embed)

        attn_mask = None
        if audio_mask is not None and text_mask is not None:
            attn_mask = createAudioTextMask(audio_mask=audio_mask, text_mask=text_mask)

        x = self.cross_attention(
            x = x,
            key = text_embed,
            mask = attn_mask,
            query_mask = audio_mask
        )

        x = self.conv_pos_embed(x, mask = audio_mask) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        num_key_value_heads = 4,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        checkpoint_activations=False,
        attn_implementation = 'default'
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.text_norm = nn.LayerNorm(text_dim)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim, heads=heads, dim_head=dim_head)
        self.input_norm = nn.LayerNorm(dim)


        self.dim = dim
        self.depth = depth
        self.attn_implementation = attn_implementation

        if attn_implementation == 'default':
            self.rotary_embed = RotaryEmbedding(dim_head)
        else:
            self.rotary_embed = ChunkLlamaRotaryEmbedding(dim = dim_head)

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(dim=dim, heads=heads, 
                         dim_head=dim_head, ff_mult=ff_mult, 
                         dropout=dropout, attn_implementation=attn_implementation, 
                         layer_idx=layer_idx, num_key_value_heads=num_key_value_heads) 
                for layer_idx in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
        audio_mask = None
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed, text_mask = self.text_embed(text, seq_len, drop_text=drop_text)
        text_embed = self.text_norm(text_embed)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, text_mask=text_mask, audio_mask=audio_mask)
        x = self.input_norm(x)

        rope = None
        if self.attn_implementation == 'default': 
            rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        position_ids = torch.arange(
            0, x.shape[1], device=x.device
        ).unsqueeze(0)

        attn_mask = audio_mask
        if self.attn_implementation == 'chunk_attn':
            attn_matrix = createAudioTextMask(audio_mask, audio_mask)
            attn_matrix = attn_matrix.unsqueeze(1) # for broadcast multi-head attention

            attn_mask = torch.zeros_like(attn_matrix, dtype = x.dtype, device = x.device)

            _MASKING_VALUE = -1e9 if x.dtype == torch.float32 else -1e4

            attn_mask = attn_mask.masked_fill(attn_matrix.unsqueeze(1) == 0, _MASKING_VALUE)


        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, attn_mask, rope, position_ids, self.rotary_embed)
            else:
                x = block(x, t, mask=attn_mask, rope=rope, position_ids=position_ids, rotary_embed=self.rotary_embed)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
