"""Correctness checks for the model. CPU only, runs in well under a minute.

    python test_model.py
"""

import torch

from model import Config, GPT, param_report, kv_cache_bytes, build_rope_cache, apply_rope

torch.manual_seed(0)


@torch.no_grad()
def generate_uncached(model, idx, n_new):
    for _ in range(n_new):
        logits, _, _ = model(idx, use_cache=False, start_pos=0)
        idx = torch.cat([idx, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, n_new):
    logits, _, caches = model(idx, use_cache=True, start_pos=0)
    pos = idx.shape[1]
    for _ in range(n_new):
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        logits, _, caches = model(nxt, use_cache=True, kv_caches=caches, start_pos=pos)
        pos += 1
    return idx


def section(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def test_kv_cache_equivalence():
    """Cached and uncached greedy decoding must agree token for token.

    This is what catches a RoPE cache that rotates by the offset within the
    current forward pass instead of the token's absolute position.
    """
    section("1. KV-cache equivalence")
    for pos_encoding in ("learned", "rope"):
        for n_kv_heads in (8, 2, 1):
            cfg = Config(n_layer=2, n_head=8, n_kv_heads=n_kv_heads, n_embd=128,
                         block_size=64, vocab_size=256, dropout=0.0,
                         pos_encoding=pos_encoding, max_infer_len=256)
            model = GPT(cfg).eval()
            prompt = torch.randint(0, 256, (2, 9))
            uncached = generate_uncached(model, prompt.clone(), 20)
            cached = generate_cached(model, prompt.clone(), 20)
            assert torch.equal(uncached, cached), (pos_encoding, n_kv_heads)
            print(f"  {pos_encoding:8s} n_kv_heads={n_kv_heads}  PASS")


def test_rope_is_relative():
    """RoPE attention scores must depend on (m - n), not on m and n separately."""
    section("2. RoPE relativity: score depends only on relative offset")
    head_dim = 64
    cos, sin = build_rope_cache(head_dim, 512, 10000.0)
    q, k = torch.randn(1, 1, 1, head_dim), torch.randn(1, 1, 1, head_dim)
    scores = []
    for m, n in [(5, 3), (105, 103), (300, 298)]:
        qr = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kr = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        score = (qr * kr).sum().item()
        scores.append(score)
        print(f"  m={m:4d} n={n:4d} offset={m - n}  score={score: .6f}")
    assert max(scores) - min(scores) < 1e-4, "scores diverge across absolute positions"
    print("  PASS")


def test_learned_positions_have_a_ceiling():
    """The learned table caps inference at block_size; RoPE does not."""
    section("3. Learned positions cap inference length, RoPE does not")
    common = dict(n_layer=2, n_head=8, n_kv_heads=8, n_embd=128,
                  block_size=64, vocab_size=256)
    learned = GPT(Config(pos_encoding="learned", **common)).eval()
    try:
        learned(torch.randint(0, 256, (1, 100)))
        raise AssertionError("expected a RuntimeError past block_size")
    except RuntimeError:
        print("  learned: RuntimeError at T=100 > block_size=64  PASS")

    rope = GPT(Config(pos_encoding="rope", max_infer_len=256, **common)).eval()
    logits, _, _ = rope(torch.randint(0, 256, (1, 100)))
    print(f"  rope:    runs at T=100, logits {tuple(logits.shape)}  PASS")


def test_parameter_budget():
    """Report the budget for each ablation row against a 600M-token schedule."""
    section("4. Parameter budget per ablation row")
    tokens = 600e6
    rows = [("baseline", "learned", 8, 4.0),
            ("rope", "rope", 8, 4.0),
            ("rope_gqa", "rope", 2, 4.0),
            ("rope_gqa_matched", "rope", 2, 4.75)]
    for name, pos, kv, mlp in rows:
        cfg = Config(pos_encoding=pos, n_kv_heads=kv, mlp_ratio=mlp)
        report = param_report(GPT(cfg), verbose=False)
        print(f"\n  [{name}]  pos={pos} n_kv_heads={kv} mlp_ratio={mlp}")
        param_report(GPT(cfg))
        print(f"  tokens/param   {tokens / report['total']:7.1f}   (Chinchilla ~20)")
        print(f"  KV cache @2048 {kv_cache_bytes(cfg, 2048) / 1e6:7.2f} MB")


def test_gqa_is_not_parameter_neutral():
    """Reducing KV heads also shrinks the k/v projections.

    Any bpb difference between an MHA and a GQA run therefore mixes two changes.
    mlp_ratio=4.75 widens the MLP to restore the original parameter count.
    """
    section("5. GQA parameter offset and the param-matched control")
    mha = sum(p.numel() for p in GPT(Config(pos_encoding="rope", n_kv_heads=8)).parameters())
    for ratio in (4.0, 4.5, 4.75, 5.0):
        cfg = Config(pos_encoding="rope", n_kv_heads=2, mlp_ratio=ratio)
        n = sum(p.numel() for p in GPT(cfg).parameters())
        flag = "  <- matched" if abs(n - mha) < 5000 else ""
        print(f"  mlp_ratio={ratio:<5} {n/1e6:6.2f}M  vs MHA {mha/1e6:.2f}M  "
              f"({(n - mha)/1e6:+.2f}M){flag}")


if __name__ == "__main__":
    test_kv_cache_equivalence()
    test_rope_is_relative()
    test_learned_positions_have_a_ceiling()
    test_parameter_budget()
    test_gqa_is_not_parameter_neutral()
    print("\nall checks passed\n")
