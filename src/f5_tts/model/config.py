import transformers
from transformers.configuration_utils import PretrainedConfig

class TTSConfig(PretrainedConfig):
    model_type = "f5_tts"

    def __init__(
            self, 
            hidden_size = 768,
            depth = 18,
            num_attention_heads = 12,
            num_key_value_heads = None,
            attn_implementation = 'chunk_attn', # default, chunk_attn
            intermediate_scale = 2,
            text_hidden_size = 512,
            conv_layers = 4,
            vocab_size = 200,
            max_position_embeddings = 131072,
            chunk_size = 2048,
            local_window = 384,
            # Mel setting
            target_sample_rate = 24000,
            n_mel_channels = 100,
            hop_length = 256,
            win_length = 1024,
            n_fft = 1024,
            mel_spec_type = 'vocos', # 'vocos' or 'bigvgan'
            # classifier-free guidance
            audio_drop_prob = 0.3,
            cond_drop_prob = 0.2,
            frac_lengths_mask = (0.7, 1.0), # for training
            # conditional flow related
            sigma = 0.0,
            odeint_method = 'euler',
            **kwargs
        ):

        # Model Parameter Settings
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_implementation = attn_implementation
        self.intermediate_scale = intermediate_scale
        self.text_hidden_size = text_hidden_size
        self.conv_layers = conv_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.chunk_size = chunk_size
        self.local_window = local_window
        # Mel setting
        self.target_sample_rate = target_sample_rate
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.mel_spec_type = mel_spec_type
        # Other param for training, inference
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.frac_lengths_mask = frac_lengths_mask
        self.sigma = sigma
        self.odeint_kwargs = {
            "method": odeint_method
        }

        assert attn_implementation in ['default', 'chunk_attn'], f"attn_implementation {attn_implementation} isn't supported. Choose: default, chunk_attn"
        assert mel_spec_type in ['vocos', 'bigvgan'], f"mel_spec_type {mel_spec_type} isn't supported. Choose: vocos, bigvgan"
        assert odeint_method in ['euler', 'midpoint'], f"odeint_method {odeint_method} isn't supported. Choose: euler, midpoint"

        super().__init__(**kwargs)