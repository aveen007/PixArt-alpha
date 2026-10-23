"""Generate and score a small PixArt-alpha T2I-CompBench baseline.

The PixArt-alpha paper reports 0.3179 on the non-spatial relationship
category of T2I-CompBench.  This tool uses the category's official
300-prompt validation split and the benchmark's CLIP ViT-B/32 cosine
similarity metric.

The default pilot generates one image per prompt.  The benchmark normally
uses multiple images per prompt, so the pilot is an alignment check rather
than an exact reproduction of the paper value.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


DATASET_ID = "Vertsineu/t2i-compbench"
DATASET_REVISION = "6c46ce25085d29158b483395fd4d7d942835c9bd"
DATASET_FILE = "non_spatial/validation-00000-of-00001.parquet"
DEFAULT_MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
PAPER_SCORE = 0.3179


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate PixArt-alpha images")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--model", default=DEFAULT_MODEL_ID)
    generate.add_argument("--start-index", type=int, default=0)
    generate.add_argument("--end-index", type=int, default=300)
    generate.add_argument("--images-per-prompt", type=int, default=1)
    generate.add_argument("--steps", type=int, default=20)
    generate.add_argument("--cfg-scale", type=float, default=4.5)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--height", type=int, default=512)
    generate.add_argument("--width", type=int, default=512)
    generate.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use model CPU offload if the complete fp16 pipeline does not fit",
    )

    score = subparsers.add_parser("score", help="Compute the official CLIP ViT-B/32 mean cosine")
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=64)

    return parser.parse_args()


def load_prompts():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    dataset_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=DATASET_FILE,
        revision=DATASET_REVISION,
        token=False,
    )
    rows = pd.read_parquet(dataset_path)
    required_columns = {"prompt_id", "prompt"}
    missing = required_columns.difference(rows.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(rows) != 300:
        raise ValueError(f"Expected 300 validation prompts, found {len(rows)}")
    return rows


def generate_images(args):
    from diffusers import DPMSolverMultistepScheduler, PixArtAlphaPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Image generation requires a CUDA GPU")
    if args.images_per_prompt < 1:
        raise ValueError("--images-per-prompt must be at least 1")

    rows = load_prompts()
    end_index = min(args.end_index, len(rows))
    if not 0 <= args.start_index < end_index:
        raise ValueError(
            f"Invalid prompt range [{args.start_index}, {args.end_index}); "
            f"the validation split has {len(rows)} prompts"
        )

    image_dir = args.output_dir / "samples"
    metadata_dir = args.output_dir / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        solver_order=2,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=0)
    else:
        pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)

    records = []
    selected = rows.iloc[args.start_index:end_index]
    for row_index, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc=f"prompts {args.start_index}:{end_index}",
    ):
        prompt = str(row["prompt"])
        prompt_id = int(row["prompt_id"])
        for sample_index in range(args.images_per_prompt):
            image_name = f"{row_index:04d}_{sample_index:02d}.png"
            image_path = image_dir / image_name
            sample_seed = args.seed + row_index * args.images_per_prompt + sample_index

            if not image_path.exists():
                generator = torch.Generator(device="cpu").manual_seed(sample_seed)
                image = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                ).images[0]
                image.save(image_path)

            records.append(
                {
                    "row_index": int(row_index),
                    "prompt_id": prompt_id,
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "prompt": prompt,
                    "image": image_name,
                    "dataset": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "model": args.model,
                    "height": args.height,
                    "width": args.width,
                    "steps": args.steps,
                    "cfg_scale": args.cfg_scale,
                }
            )

    metadata_path = metadata_dir / f"prompts_{args.start_index:04d}_{end_index:04d}.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated/resumed {len(records)} image records")
    print(f"Metadata: {metadata_path}")


def load_generation_records(output_dir):
    records_by_image = {}
    metadata_paths = sorted((output_dir / "metadata").glob("prompts_*.jsonl"))
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata shards found under {output_dir / 'metadata'}")

    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records_by_image[record["image"]] = record

    records = sorted(
        records_by_image.values(),
        key=lambda record: (record["row_index"], record["sample_index"]),
    )
    missing_images = [
        record["image"]
        for record in records
        if not (output_dir / "samples" / record["image"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} generated images; first: {missing_images[0]}")
    return records


def score_images(args):
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI CLIP is required. Install it with: "
            "python -m pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    records = load_generation_records(args.output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    scored_records = []
    for offset in tqdm(range(0, len(records), args.batch_size), desc="CLIP scoring"):
        batch = records[offset : offset + args.batch_size]
        images = torch.stack(
            [preprocess(Image.open(args.output_dir / "samples" / record["image"]).convert("RGB")) for record in batch]
        ).to(device)
        texts = clip.tokenize([record["prompt"] for record in batch]).to(device)

        with torch.inference_mode():
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features * text_features).sum(dim=-1).float().cpu().tolist()

        for record, similarity in zip(batch, similarities):
            scored_records.append({**record, "clip_vit_b32_cosine": similarity})

    scores = torch.tensor([record["clip_vit_b32_cosine"] for record in scored_records])
    prompt_counts = {}
    for record in scored_records:
        row_index = record["row_index"]
        prompt_counts[row_index] = prompt_counts.get(row_index, 0) + 1

    summary = {
        "metric": "T2I-CompBench non-spatial CLIP ViT-B/32 cosine",
        "num_images": len(scored_records),
        "num_prompts": len({record["row_index"] for record in scored_records}),
        "images_per_prompt": sorted(set(prompt_counts.values())),
        "complete_validation_split": len(prompt_counts) == 300,
        "mean": scores.mean().item(),
        "std": scores.std(unbiased=False).item(),
        "paper_pixart_alpha": PAPER_SCORE,
        "difference_from_paper": scores.mean().item() - PAPER_SCORE,
        "note": "One-image-per-prompt runs are pilot estimates, not exact paper reproductions.",
    }

    results_path = args.output_dir / "clip_scores.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in scored_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = args.output_dir / "clip_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Per-image scores: {results_path}")
    print(f"Summary: {summary_path}")


def main():
    args = parse_args()
    if args.command == "generate":
        generate_images(args)
    else:
        score_images(args)


if __name__ == "__main__":
    main()
