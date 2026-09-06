# GPT-2 baseline -> RoPE -> GQA: architecture ablation

8-layer, 8-head, 512-dim decoder implemented from scratch and trained on TinyStories
(466.75M tokens, custom 8192 BPE). Each row changes exactly one variable; all rows share
the same tokenizer, batch order, seed and 36,621-iteration schedule.

Metric is **bits-per-byte**, not perplexity: the vocabulary changed from 50257 to 8192,
so per-token perplexity is not comparable across configurations.

## Rows

| run | positions | KV heads | params | val bpb | KV cache @2048 | max context |
|---|---|---|---|---|---|---|
| `baseline` | learned | 8 | 29.68M | **0.4814** | 33.55 MB | 512 (hard cap) |
| `rope` | rope | 8 | 29.41M | **0.4733** | 33.55 MB | 4096 |
| `rope_gqa` | rope | 2 | 26.26M | **0.4783** | 8.39 MB | 4096 |

## Findings

**RoPE.** bpb 0.4814 -> 0.4733 (-1.68%) while removing 262K parameters.
More importantly it removes a hard inference ceiling: the learned-embedding baseline
holds 512 positions and raises a RuntimeError at position 513, so it cannot be
benchmarked at 1024 or 2048 context at all.

**GQA.** 8 -> 2 KV heads gives an exact 4x cache reduction (33.6 MB -> 8.4 MB at 2048) for +1.1% bpb, plus 11% fewer parameters
from the smaller k/v projections.

## Where GQA starts paying off

Decode throughput swept over batch and context. Speedup is GQA steps/s over MHA steps/s.

| batch | ctx | MHA cache | GQA cache | MHA steps/s | GQA steps/s | GQA speedup |
|---|---|---|---|---|---|---|
| 1 | 512 | 0.01 GB | 0.00 GB | 170.7 | 140.7 | **0.82x** |
| 1 | 2048 | 0.03 GB | 0.01 GB | 177.3 | 138.6 | **0.78x** |
| 16 | 512 | 0.13 GB | 0.03 GB | 152.5 | 143.2 | **0.94x** |
| 16 | 2048 | 0.54 GB | 0.13 GB | 48.6 | 64.5 | **1.33x** |
| 64 | 512 | 0.54 GB | 0.13 GB | 46.7 | 60.4 | **1.29x** |
| 64 | 2048 | 2.15 GB | 0.54 GB | 13.2 | 18.0 | **1.36x** |

GQA is a memory-bandwidth intervention, not a compute one, so it only helps once the
KV cache is actually the bottleneck. At batch 1 the cache is 0.03 GB against ~59 MB of
weights: decode is weight-bound, GQA relieves a bottleneck that is not binding, and its
extra kernel overhead makes it a net **0.78x**. By batch 64 at 2048 context the MHA cache
reaches 2.15 GB and GQA reaches **1.36x**.

Speedup tracks cache size rather than batch or context on their own. The two
configurations where the MHA cache is both 0.54 GB -- batch 16 at ctx 2048, and batch 64
at ctx 512 -- give 1.33x and 1.29x despite differing 4x in both batch and context.

## Caveats

- baseline ran before the batch-order fix, so its bpb carries a small data-order confound
- rope_gqa has 3.15M fewer params than rope (smaller k/v projections); the param-matched control (mlp_ratio 4.75) was not run
- 600M tokens over a 466.75M-token corpus = 1.29 epochs, identical across rows

## Reproducing

`notebooks/01_prepare_data.ipynb` builds the tokenizer and token bins; `notebooks/02_train.ipynb` trains
one row (set `RUN`); `notebooks/03_collect_results.ipynb` assembles the table. Data files are not
redistributed here -- TinyStories is CDLA-Sharing-1.0, so the preparation notebook
rebuilds them from the source dataset.
