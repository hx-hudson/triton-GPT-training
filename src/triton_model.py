import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from dataclasses import dataclass
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from triton_kernel.layernorm import LayerNorm_triton
from triton_kernel.flash_attention import flash_attention_triton



class MlP(nn.Module):

    def __init__(self, embed_size):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, 4 * embed_size)
        self.linear2 = nn.Linear(embed_size * 4, embed_size)
        self.linear2.SCALE_INIT = 1 # type: ignore

    def forward(self, x):

        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)

        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_size, head_nums):
        super().__init__()
        self.head_size = embed_size // head_nums
        self.head_nums = head_nums
        self.qkv_weight = nn.Linear(embed_size, embed_size * 3, bias=False)
        self.linear = nn.Linear(embed_size, embed_size, bias=False)
        self.linear.SCALE_INIT = 1 # type: ignore

    def forward(self, x: torch.Tensor):

        B, T, C = x.shape
        qkv = self.qkv_weight(x) # (B, T, (Q + K + V) * head_nums)
        qkv = qkv.view(B, T, 3, self.head_nums, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = flash_attention_triton(q, k, v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(B, T, C)

        x = self.linear(x)
        return x

class Block(nn.Module):

    def __init__(self, embed_size, head_nums):
        super().__init__()
        self.mlp = MlP(embed_size)
        self.multi_attention = MultiHeadAttention(embed_size, head_nums)
        self.ln1 = LayerNorm_triton(embed_size)
        self.ln2 = LayerNorm_triton(embed_size)

    def forward(self, x):

        x = x + self.multi_attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

@dataclass
class GPTConfig:
    token_nums: int = 50257
    embed_size: int = 768
    head_nums: int = 12
    window_size: int = 1024
    block_nums: int = 12


class triton_GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.window_size = config.window_size
        self.block_nums = config.block_nums
        self.token_embed = nn.Embedding(config.token_nums, config.embed_size)
        self.pos_embed = nn.Embedding(config.window_size, config.embed_size)
        self.blocks = nn.ModuleList([
            Block(config.embed_size, config.head_nums) for _ in range(config.block_nums)
        ])
        self.last_linear = nn.Linear(config.embed_size, config.token_nums, bias=False)

        self.ln = nn.LayerNorm(config.embed_size)
        
        self.apply(self._init_weight)
        self.last_linear.weight = self.token_embed.weight

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.block_nums) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.token_embed(x) + self.pos_embed(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.last_linear(x)

        return logits