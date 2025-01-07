import math
import torch
from torch import nn
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func

chunk_size = None
local_window = None
MAX_NEW_TOKENS = 512


def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.05 * math.log(scale) + 1.0


def apply_rotary_pos_emb(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_emb = (x * cos) + (rotate_half(x) * sin)
    return x_emb


def merge_attn_outputs(flash_results):
    attn_outputs_all = [flash_results[0][0]]
    flash_results = flash_results[1:]
    for flash_per_chunk in flash_results:
        attn_outputs = torch.stack([flash_attn_output[0] for flash_attn_output in flash_per_chunk])
        logits = torch.stack([flash_attn_output[1] for flash_attn_output in flash_per_chunk]).to(torch.float32)
        max_logits = torch.max(logits, dim=0).values
        stable_logits = logits - max_logits.unsqueeze(0)

        lse_s = torch.exp(stable_logits).detach()
        lse_sum = torch.sum(lse_s, dim=0)
        lse_s /= lse_sum
        lse_s = lse_s.to(torch.bfloat16)
        attn_outputs *= lse_s.unsqueeze(-1)
        attn_outputs_all.append(attn_outputs.sum(dim=0))
    return torch.cat(attn_outputs_all, dim=2)

def eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    causal: bool = False
):
    heads, head_dim = query_states.size(1), query_states.size(3)
    scaling = (heads * head_dim) ** -0.5

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if causal is True:
        L, S = query_states.size(-2), key_states.size(-2)
        attn_bias = torch.zeros(L, S, dtype=query_states.dtype, device = query_states.device)
        temp_mask = torch.ones(L, S, dtype=torch.bool, device = query_states.device).tril(diagonal=0)

        _MASKING_VALUE = -1e9 if query_states.dtype == torch.float32 else -1e4

        attn_bias.masked_fill_(temp_mask.logical_not(), _MASKING_VALUE)
        attn_bias.to(query_states.dtype)

        attn_weights += attn_bias
        
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def supports_flash_attention(device_id):
    """Check if a GPU supports FlashAttention."""
    if not torch.cuda.is_available(): return False

    major, minor = torch.cuda.get_device_capability(device_id)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90

def do_flash_attn(query_states, key_states, value_states, causal=True):
    # flash_attention
    if supports_flash_attention(query_states.device):
        output, softmax_lse, _ = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2),
                                                value_states.transpose(1, 2), causal=causal, return_attn_probs=True)
    else:
        output, softmax_lse = eager_attention_forward(query_states, key_states, value_states, causal = causal)

    return output.transpose(1, 2), softmax_lse


class ChunkLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, scaling_factor=1.0, device=None):
        super().__init__()

        self.max_seq_len = max_position_embeddings
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.chunk_size = max_position_embeddings * 3 // 4
        self.local_window = max_position_embeddings // 8
        self.chunk_len = self.chunk_size - self.local_window

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_seq_len,
            device=device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # employing yarn will lead to better performance but results reported in our paper did not use yarn.
        scale = seq_len / self.max_position_embeddings
        mscale = get_mscale(scale)
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        q_t = torch.arange(self.chunk_len, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
        qc_t = (torch.arange(self.chunk_len, device=device, dtype=self.inv_freq.dtype) + self.chunk_len).clamp(
            max=self.chunk_size) / self.scaling_factor
        k_t = (torch.arange(seq_len + MAX_NEW_TOKENS, device=device,
                            dtype=self.inv_freq.dtype) % self.chunk_len) / self.scaling_factor

        q_freqs = torch.outer(q_t, self.inv_freq)  # seq_len x dim/2
        qc_freqs = torch.outer(qc_t, self.inv_freq)
        k_freqs = torch.outer(k_t, self.inv_freq)  # seq_len x dim/2

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        q_emb = torch.cat((q_freqs, q_freqs), dim=-1)  # seq_len x dim
        qc_emb = torch.cat((qc_freqs, qc_freqs), dim=-1)
        k_emb = torch.cat((k_freqs, k_freqs), dim=-1)  # seq_len x dim
        self.register_buffer("q_cos_cached", q_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("q_sin_cached", q_emb.sin().to(dtype) * mscale, persistent=False)
        self.register_buffer("qc_cos_cached", qc_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("qc_sin_cached", qc_emb.sin().to(dtype) * mscale, persistent=False)
        self.register_buffer("k_cos_cached", k_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("k_sin_cached", k_emb.sin().to(dtype) * mscale, persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # no token will exceed chunk_size
        # chunk1_q,
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device, dtype=torch.float32)
            self.max_seq_len = seq_len
        return (
            self.q_cos_cached[:seq_len].to(dtype=x.dtype),
            self.q_sin_cached[:seq_len].to(dtype=x.dtype),
            self.qc_cos_cached[:seq_len].to(dtype=x.dtype),
            self.qc_sin_cached[:seq_len].to(dtype=x.dtype),
            self.k_cos_cached[:seq_len].to(dtype=x.dtype),
            self.k_sin_cached[:seq_len].to(dtype=x.dtype),
        )
    

class ChunkLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 hidden_size: int, 
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 layer_idx: int,
                 attention_dropout: float = 0.0,
                 max_position_embeddings: int = 32768,
                 chunk_size=2048,
                 local_window=384
                ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=False
        )

        self.chunk_size = chunk_size
        self.local_window = local_window
        self.chunk_len = self.chunk_size - self.local_window

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            rotary_embed: nn.Module = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # during inference
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        q_seq_len = query_states.shape[-2]
        has_kv_cache = q_seq_len != kv_seq_len
        # covert to b x head x len x h
        # need to chunk query states
        q_cos, q_sin, qc_cos, qc_sin, k_cos, k_sin = rotary_embed(value_states, seq_len=kv_seq_len)
        key_states = apply_rotary_pos_emb(key_states, k_cos, k_sin, position_ids)
        position_ids = position_ids % self.chunk_len

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=None)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        flash_results = []
        if not has_kv_cache:
            q_states_intra = apply_rotary_pos_emb(query_states[:, :, :self.chunk_len, :], q_cos, q_sin,
                                                position_ids[:, :self.chunk_len])
            k_states_prev = key_states[:, :, :self.chunk_len, :]
            v_states_prev = value_states[:, :, :self.chunk_len, :]
            flash_results.append(do_flash_attn(q_states_intra, k_states_prev, v_states_prev))
            remain_len = kv_seq_len - self.chunk_len

            while remain_len > 0:
                flash_per_chunk = []
                begin = kv_seq_len - remain_len
                curr_chunk_len = min(self.chunk_len, remain_len)
                end = begin + curr_chunk_len

                q_states_intra = apply_rotary_pos_emb(query_states[:, :, begin:end, :], q_cos, q_sin,
                                                    position_ids[:, begin:end])

                k_states_intra = key_states[:, :, begin:end, :]
                v_states_intra = value_states[:, :, begin:end, :]
                flash_per_chunk.append(do_flash_attn(q_states_intra, k_states_intra, v_states_intra))

                q_states_succ = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                    position_ids[:, begin:end])
                flash_per_chunk.append(do_flash_attn(q_states_succ, k_states_prev, v_states_prev, False))

                if begin - (k_states_prev.size(-2)) > 0:
                    prev_len = k_states_prev.size(-2)
                    q_states_inter = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                        position_ids[:, self.chunk_len - 1][:, None].repeat(1, curr_chunk_len))
                    k_states_inter = key_states[:, :, :begin - prev_len, :]
                    v_states_inter = value_states[:, :, :begin - prev_len, :]
                    flash_per_chunk.append(do_flash_attn(q_states_inter, k_states_inter, v_states_inter, False))

                flash_results.append(flash_per_chunk)
                k_states_prev = k_states_intra
                v_states_prev = v_states_intra
                remain_len = remain_len - self.chunk_len

            attn_output = merge_attn_outputs(flash_results)
        else:
            chunk_num_curr = (kv_seq_len - 1) // self.chunk_len
            q_states_intra = apply_rotary_pos_emb(query_states, q_cos, q_sin, position_ids)
            k_states_intra = key_states[:, :, self.chunk_len * chunk_num_curr:kv_seq_len, :]
            attn_weights = torch.matmul(q_states_intra, k_states_intra.transpose(2, 3)) / math.sqrt(
                self.head_dim)
            attn_scores = [attn_weights]

            if chunk_num_curr >= 1:
                q_states_succ = apply_rotary_pos_emb(query_states, qc_cos, qc_sin, position_ids)

                k_states_succ = key_states[:, :, self.chunk_len * (chunk_num_curr - 1):self.chunk_len * chunk_num_curr, :]
                attn_weights = torch.matmul(q_states_succ, k_states_succ.transpose(2, 3)) / math.sqrt(
                    self.head_dim)
                attn_scores = [attn_weights] + attn_scores

            if chunk_num_curr >= 2:
                q_states_inter = apply_rotary_pos_emb(query_states, qc_cos, qc_sin,
                                                    torch.tensor([[self.chunk_len - 1]], device=query_states.device))
                k_states_inter = key_states[:, :, :self.chunk_len * (chunk_num_curr - 1), :]
                attn_weights = torch.matmul(q_states_inter, k_states_inter.transpose(2, 3)) / math.sqrt(
                    self.head_dim)
                attn_scores = [attn_weights] + attn_scores

            attn_weights = torch.cat(attn_scores, dim=-1)
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value