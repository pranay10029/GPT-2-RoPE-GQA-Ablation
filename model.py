"""Decoder-only transformer with switchable positional encoding and KV-head count.

A single Config drives every ablation row:

    row                pos_encoding   n_kv_heads   mlp_ratio
    baseline           learned        8            4.0
    rope               rope           8            4.0
    rope_gqa           rope           2            4.0
    rope_gqa_matched   rope           2            4.75   (param-matched to rope)
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sdpa_supports_gqa() -> bool:
    """scaled_dot_product_attention is a builtin; probe rather than introspect."""
    try:
        q = torch.zeros(1, 4, 2, 8)
        k = torch.zeros(1, 2, 2, 8)
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        return True
    except Exception:
        return False


SDPA_GQA = _sdpa_supports_gqa()


@dataclass
class Config:
    n_layer: int = 8
    n_head: int = 8
    n_kv_heads: int = 8
    n_embd: int = 512
    block_size: int = 512
    vocab_size: int = 8192
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pos_encoding: str = "learned"
    rope_theta: float = 10000.0
    max_infer_len: int = 4096

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_heads == 0
        assert self.pos_encoding in ("learned", "rope")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def build_rope_cache(head_dim, max_seq, theta, device=None, dtype=torch.float32):
    """Precompute (cos, sin), each of shape (max_seq, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq, device=device).float(), inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate x by the positions encoded in cos/sin.

    x        -- (B, n_head, T, head_dim)
    cos, sin -- (T, head_dim // 2), pre-sliced to absolute positions

    Split-half layout (GPT-NeoX/HF), applied identically to q and k. Accumulated
    in fp32: under fp16 autocast the rotation loses precision at long positions.
    """
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)


class CausalSelfAttention(nn.Module):
    """Multi-head attention; n_kv_heads < n_head selects grouped-query attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_heads
        self.dropout = cfg.dropout

        self.q_dim = cfg.n_head * cfg.head_dim
        self.kv_dim = cfg.n_kv_heads * cfg.head_dim

        self.qkv = nn.Linear(cfg.n_embd, self.q_dim + 2 * self.kv_dim)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        # Full-sequence prefill or single-token decode only. Chunked prefill against
        # a populated cache would need an explicit mask, since is_causal aligns
        # top-left rather than bottom-right when q_len != k_len.
        assert kv_cache is None or T == 1

        q, k, v = self.qkv(x).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # v unrotated

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if use_cache else None

        is_causal = T > 1
        p = self.dropout if self.training else 0.0

        if self.n_rep == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=p)
        elif SDPA_GQA:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal,
                                               dropout_p=p, enable_gqa=True)
        else:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        self.fc1 = nn.Linear(cfg.n_embd, hidden)
        self.fc2 = nn.Linear(hidden, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln1(x), rope, kv_cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_rope = cfg.pos_encoding == "rope"

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if self.use_rope:
            cos, sin = build_rope_cache(cfg.head_dim, cfg.max_infer_len, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.pos_emb = None
        else:
            # Sized to block_size exactly, so the inference ceiling is real and
            # measurable rather than padded around.
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None, use_cache=False, start_pos=0):
        B, T = idx.shape
        x = self.tok_emb(idx)

        rope = None
        if self.use_rope:
            # start_pos is the absolute index of idx[:, 0]. During cached decode
            # T == 1 and start_pos advances; slicing from 0 instead trains fine
            # and generates nonsense.
            assert start_pos + T <= self.rope_cos.size(0), "increase max_infer_len"
            rope = (self.rope_cos[start_pos:start_pos + T],
                    self.rope_sin[start_pos:start_pos + T])
        else:
            if start_pos + T > self.pos_emb.num_embeddings:
                raise RuntimeError(
                    f"learned positional table holds {self.pos_emb.num_embeddings} "
                    f"positions, requested {start_pos + T}"
                )
            x = x + self.pos_emb(torch.arange(start_pos, start_pos + T, device=idx.device))

        x = self.drop(x)

        new_caches = []
        for i, block in enumerate(self.blocks):
            x, nc = block(x, rope, kv_caches[i] if kv_caches else None, use_cache)
            new_caches.append(nc)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        else:
            logits = self.head(x[:, [-1], :])  # only the last position is needed
            loss = None

        return logits, loss, (new_caches if use_cache else None)


def param_report(model: GPT, verbose=True):
    """Split the parameter count into embedding and transformer components."""
    total = sum(p.numel() for p in model.parameters())  # tied weights counted once
    emb = model.tok_emb.weight.numel()
    pos = model.pos_emb.weight.numel() if model.pos_emb is not None else 0
    non_emb = total - emb - pos
    if verbose:
        print(f"  total          {total/1e6:7.2f}M")
        print(f"  token emb      {emb/1e6:7.2f}M  ({100*emb/total:.1f}%)")
        print(f"  pos emb        {pos/1e6:7.2f}M")
        print(f"  transformer    {non_emb/1e6:7.2f}M")
    return {"total": total, "emb": emb, "pos": pos, "non_emb": non_emb}


def kv_cache_bytes(cfg: Config, seq_len, batch=1, dtype_bytes=2):
    """2 (K and V) x layers x kv_heads x head_dim x seq_len x batch x bytes."""
    return 2 * cfg.n_layer * cfg.n_kv_heads * cfg.head_dim * seq_len * batch * dtype_bytes
