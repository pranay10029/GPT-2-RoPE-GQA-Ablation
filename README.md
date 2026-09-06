# GPT-2 from Scratch: RoPE and GQA Ablations

An 8-layer, 8-head, 512-dim autoregressive decoder implemented from first principles in
PyTorch, trained on TinyStories, then used as a controlled baseline to measure two modern
architectural components: **rotary position embeddings (RoPE)** and **grouped-query
attention (GQA)**.

The point is not the model. It is the measurement: each row changes exactly one variable,
and the headline result is a decode-throughput sweep locating the batch/context regime
where GQA starts paying off — and showing it *costs* throughput below that point.

## Results

| run | positions | KV heads | params | val bpb | KV cache @2048 | max context |
|---|---|---|---|---|---|---|
| `baseline` | learned | 8 | 29.68M | **0.4814** | 33.55 MB | 512 (hard cap) |
| `rope` | RoPE | 8 | 29.41M | **0.4733** | 33.55 MB | 4096 |
| `rope_gqa` | RoPE | 2 | 26.26M | **0.4783** | 8.39 MB | 4096 |

### Where GQA starts paying off

| batch | ctx | MHA cache | GQA cache | GQA speedup |
|---|---|---|---|---|
| 1 | 512 | 0.01 GB | 0.00 GB | **0.82x** |
| 1 | 2048 | 0.03 GB | 0.01 GB | **0.78x** |
| 16 | 512 | 0.13 GB | 0.03 GB | **0.94x** |
| 16 | 2048 | 0.54 GB | 0.13 GB | **1.33x** |
| 64 | 512 | 0.54 GB | 0.13 GB | **1.29x** |
| 64 | 2048 | 2.15 GB | 0.54 GB | **1.36x** |

GQA is a memory-bandwidth intervention. At batch 1 the KV cache is 0.03 GB against ~59 MB
of weights, so decode is weight-bound, GQA relieves a bottleneck that is not binding, and
its extra kernel overhead makes it a net **0.78x**. By batch 64 at 2048 context the MHA
cache reaches 2.15 GB and GQA reaches **1.36x**.

Speedup tracks *cache size*, not batch or context individually: the two configurations
where the MHA cache is both 0.54 GB — batch 16 at ctx 2048, and batch 64 at ctx 512 — give
1.33x and 1.29x despite differing 4x in both batch and context.

Full write-up and caveats: [RESULTS.md](RESULTS.md). Raw numbers: [benchmark.json](benchmark.json).

## Setup

- 466.75M training tokens (TinyStories), custom **8192-vocab byte-level BPE**
- 600M tokens seen over 36,621 iterations = ~20 tokens/param (Chinchilla-optimal)
- batch 32 x 512 context, AdamW + warmup/cosine, fp16 AMP, grad clip 1.0
- Tesla T4, PyTorch 2.10, ~2.6 h per row

Metric is **bits-per-byte**, not perplexity. The vocabulary changed from 50257 to 8192, so
per-token perplexity is not comparable across configurations.

## Design notes

**One file, flags, not four model files.** `pos_encoding` and `n_kv_heads` in `Config`
select the row. Separate per-variant files rot: a batching fix lands in one and not the
others, and the rows quietly stop being comparable.

**Batch order needs its own RNG.** Drawing batches from the global RNG was a real bug here.
Each config consumes a different amount of RNG during init — RoPE has no position table,
GQA has smaller k/v projections — so every row silently trained on a *different* data
stream. `DATA_SEED` is seeded after model construction to fix this.

**RoPE must use absolute positions during cached decode.** Slicing cos/sin from 0 on every
single-token step is the classic failure: training looks perfect and generation degrades
into nonsense. `test_model.py` catches it by asserting cached and uncached greedy
generation produce identical tokens.

**The learned-position table is exactly `block_size`.** No `block_size + 200` padding hack.
The baseline raises a RuntimeError at position 513, which is the RoPE result measured
rather than asserted.

## Reproducing

```bash
python test_model.py    # correctness only, CPU, ~30s
```

Then:

1. `notebooks/01_prepare_data.ipynb` — builds tokenizer and token bins (once, CPU, ~25 min)
2. `notebooks/02_train.ipynb` — trains one row; set `RUN`, repeat per config (~2.6 h each)
3. `notebooks/03_collect_results.ipynb` — assembles the comparison table

`notebooks/02_train_colab.ipynb` is the same as (2) with Google Drive checkpointing.

Token bins are not redistributed here: TinyStories is CDLA-Sharing-1.0. Notebook (1)
rebuilds them from the source dataset. The tokenizer trained for these runs is in
`tokenizer/` (sha256[:16] `87302aa2fe41fad2`) so results stay comparable.

## Caveats

- `baseline` ran before the batch-order fix, so its bpb carries a small data-order confound
- `rope_gqa` has 3.15M fewer params than `rope` (smaller k/v projections); the
  param-matched control (`mlp_ratio=4.75`, restoring 29.41M) was not run, so the +1.1% bpb
  is not attributable to the KV-head change alone
- 600M tokens over a 466.75M-token corpus = 1.29 epochs, identical across rows

## Layout

```
model.py                  Config, RoPE, GQA attention, GPT, param/cache accounting
test_model.py             cache-equivalence, RoPE relativity, param budget checks
RESULTS.md                full write-up
benchmark.json            all measured numbers
notebooks/                data prep, training, collection
results/                  per-run results + generation samples
tokenizer/                the exact 8192 BPE used for every row
```

## Credits

Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(Eldan & Li, 2023), CDLA-Sharing-1.0.
RoPE: Su et al., 2021. GQA: Ainslie et al., 2023.

## License

MIT (code). See `tokenizer/` note above regarding dataset licensing.
