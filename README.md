# PixArt-alpha with projected LLaVA conditioning

This fork evaluates PixArt-alpha with LLaVA token embeddings projected into PixArt T5 conditioning space.

Original installation, checkpoints, training, and inference documentation: [README (base repo).md](README%20%28base%20repo%29.md).

## Workflow

### 1. Export native PixArt T5 conditioning

[`tools/export_t2i_compbench_text_embeddings.py`](tools/export_t2i_compbench_text_embeddings.py) exports the 300 T2I-CompBench prompts and 1,200 additional T5 vocabulary targets.

```bash
python tools/export_t2i_compbench_text_embeddings.py \
  --output native_t5_active_tokens_plus_1200_vocab_fp16.pt \
  --extra-vocab-tokens 1200 \
  --batch-size 4
```

### 2. Extract LLaVA embeddings and fit the projection

Run [`notebooks/pixart-book.ipynb`](notebooks/pixart-book.ipynb) on Kaggle with GPU and Internet enabled.

Required Kaggle input:

- `native_t5_active_tokens_plus_1200_vocab_fp16.pt`

Notebook outputs:

- `pixart_llava_compbench_plus_1200_vocab_seam_pairs.pt`
- `pixart_llava_compbench_plus_1200_vocab_projector.pt`
- `pixart_llava_compbench_projected_plus_1200_vocab_active_fp16.pt`

Copy the final projected conditioning file to the repository root.

## Evaluation

Evaluation code: [`tools/eval_t2i_compbench_nonspatial.py`](tools/eval_t2i_compbench_nonspatial.py).

Native PixArt baseline:

```bash
python tools/eval_t2i_compbench_nonspatial.py generate \
  --output-dir output/pixart_compbench_nonspatial

python tools/eval_t2i_compbench_nonspatial.py score \
  --output-dir output/pixart_compbench_nonspatial
```

Projected LLaVA conditioning:

```bash
python tools/eval_t2i_compbench_nonspatial.py generate \
  --output-dir output/pixart_compbench_nonspatial_projected_plus_1200_vocab \
  --projected-text-embeddings pixart_llava_compbench_projected_plus_1200_vocab_active_fp16.pt

python tools/eval_t2i_compbench_nonspatial.py score \
  --output-dir output/pixart_compbench_nonspatial_projected_plus_1200_vocab
```

Generated images are saved under `<output-dir>/samples/`. Scores are saved in `<output-dir>/clip_summary.json` and `<output-dir>/clip_scores.jsonl`.

## Results

Full settings and checksums: [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md).

| Run | CLIP ViT-B/32 cosine |
|---|---:|
| PixArt-alpha paper | 0.3179 |
| Local native-T5 baseline | 0.3098 |
| LLaVA projected | 0.2984 |
| LLaVA projected + 1,200 T5 classes | 0.2999 |

`output/` and generated `.pt` files are ignored by Git. The two evaluated projected conditioning files retained in the repository root are:

- `pixart_llava_compbench_projected_active_fp16.pt`
- `pixart_llava_compbench_projected_plus_1200_vocab_active_fp16.pt`
