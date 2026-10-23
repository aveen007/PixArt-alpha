# PixArt-alpha baseline results

## T2I-CompBench non-spatial relationships

| Run | Prompts | Images per prompt | CLIP ViT-B/32 cosine (mean) | Standard deviation | Difference from paper | Relative difference |
|---|---:|---:|---:|---:|---:|---:|
| PixArt-alpha paper | 300 | Multiple (official protocol) | **0.3179** | Not reported | N/A | N/A |
| Local baseline pilot | 300 | 1 | **0.3098** | 0.0273 | -0.0081 | -2.55% |

The local baseline is close to the reported result, but it is not an exact
reproduction. It uses one generated image per prompt, whereas the official
T2I-CompBench protocol uses multiple images per prompt. The result should be
treated as the reference baseline for later projection experiments run with
the same script and settings.

### Run details

| Setting | Value |
|---|---|
| Date | 2026-10-24 |
| Model | `PixArt-alpha/PixArt-XL-2-512x512` |
| Model revision | `50f702106901db6d0f8b67eb88e814c56ded2692` |
| Dataset | `Vertsineu/t2i-compbench`, non-spatial validation split |
| Dataset revision | `6c46ce25085d29158b483395fd4d7d942835c9bd` |
| Resolution | 512 x 512 |
| Sampler | DPM-Solver Multistep, solver order 2 |
| Denoising steps | 20 |
| Classifier-free guidance | 4.5 |
| Base seed | 0 |
| Images evaluated | 300 |
| Generation time | 6 minutes 36 seconds |
| Evaluation script | `tools/eval_t2i_compbench_nonspatial.py` |
| Saved summary | `output/pixart_compbench_nonspatial/clip_summary.json` |

The Diffusers loader reported that `caption_projection.y_embedding` was not
used. This is another reason to regard this result as a lightweight reference
baseline rather than an exact reproduction of the original implementation.

Paper reference: [PixArt-alpha: Fast Training of Diffusion Transformer for
Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426)
(T2I-CompBench non-spatial score: 0.3179).
