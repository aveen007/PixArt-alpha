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
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


DATASET_ID = "Vertsineu/t2i-compbench"
DATASET_REVISION = "6c46ce25085d29158b483395fd4d7d942835c9bd"
DATASET_FILE = "non_spatial/validation-00000-of-00001.parquet"
DEFAULT_MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"
DEFAULT_MODEL_REVISION = "50f702106901db6d0f8b67eb88e814c56ded2692"
PAPER_SCORE = 0.3179
NATIVE_ONE_IMAGE_BASELINE_SCORE = 0.3098067343235016
PROJECTED_CONDITIONING_FORMAT = "pixart_t2i_compbench_projected_active_conditioning_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate PixArt-alpha images")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--model", default=DEFAULT_MODEL_ID)
    generate.add_argument(
        "--projected-text-embeddings",
        type=Path,
        help=(
            "Compact projected active-token bundle. When provided, the T5 tokenizer and "
            "encoder are not loaded."
        ),
    )
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


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_projected_conditioning(path, rows, model_id):
    if not path.is_file():
        raise FileNotFoundError(f"Projected conditioning file not found: {path}")

    bundle = torch.load(path, map_location="cpu", weights_only=True)
    required_keys = {
        "format",
        "active_prompt_embeds",
        "prompt_row_offsets",
        "active_prompt_positions",
        "active_prompt_input_ids",
        "prompt_attention_mask",
        "prompt_input_ids",
        "active_negative_prompt_embeds",
        "negative_prompt_row_offsets",
        "active_negative_prompt_positions",
        "active_negative_prompt_input_ids",
        "negative_prompt_attention_mask",
        "negative_prompt_input_ids",
        "prompts",
        "prompt_sha256",
        "row_indices",
        "prompt_ids",
        "metadata",
    }
    missing = required_keys.difference(bundle)
    if missing:
        raise ValueError(f"Projected conditioning is missing keys: {sorted(missing)}")
    if bundle["format"] != PROJECTED_CONDITIONING_FORMAT:
        raise ValueError(
            f"Expected projected format {PROJECTED_CONDITIONING_FORMAT!r}, "
            f"found {bundle['format']!r}"
        )

    metadata = bundle["metadata"]
    expected_prompts = [str(prompt) for prompt in rows["prompt"].tolist()]
    expected_prompt_ids = torch.tensor(rows["prompt_id"].tolist(), dtype=torch.int64)
    expected_hashes = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in expected_prompts
    ]
    if bundle["prompts"] != expected_prompts:
        raise ValueError("Projected prompt text/order does not match the pinned evaluation split")
    if bundle["prompt_sha256"] != expected_hashes:
        raise ValueError("Projected prompt hashes do not match the pinned evaluation split")
    if not torch.equal(bundle["row_indices"].long(), torch.arange(len(rows))):
        raise ValueError("Projected row indices are not exactly 0..299")
    if not torch.equal(bundle["prompt_ids"].long(), expected_prompt_ids):
        raise ValueError("Projected prompt IDs do not match the pinned evaluation split")

    if metadata.get("model") != model_id:
        raise ValueError(
            f"Projected model {metadata.get('model')!r} does not match requested model {model_id!r}"
        )
    if model_id == DEFAULT_MODEL_ID and metadata.get("model_revision") != DEFAULT_MODEL_REVISION:
        raise ValueError("Projected conditioning was produced for a different model revision")
    if metadata.get("dataset") != DATASET_ID or metadata.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Projected conditioning was produced for a different dataset revision")
    if metadata.get("logical_prompt_embedding_shape") != [len(rows), 120, 4096]:
        raise ValueError("Projected conditioning has unexpected logical dimensions")
    if metadata.get("native_cfg_negative_preserved") is not True:
        raise ValueError("Projected conditioning does not declare a preserved native CFG negative")

    attention_mask = bundle["prompt_attention_mask"].long()
    input_ids = bundle["prompt_input_ids"].long()
    active_embeddings = bundle["active_prompt_embeds"]
    active_positions = bundle["active_prompt_positions"].long()
    active_input_ids = bundle["active_prompt_input_ids"].long()
    row_offsets = bundle["prompt_row_offsets"].long()
    expected_dense_shape = (len(rows), 120)
    if tuple(attention_mask.shape) != expected_dense_shape:
        raise ValueError(f"Expected prompt mask shape {expected_dense_shape}, got {tuple(attention_mask.shape)}")
    if tuple(input_ids.shape) != expected_dense_shape:
        raise ValueError(f"Expected prompt ID shape {expected_dense_shape}, got {tuple(input_ids.shape)}")
    if active_embeddings.ndim != 2 or active_embeddings.shape[1] != 4096:
        raise ValueError(f"Expected active embeddings shaped [tokens, 4096], got {tuple(active_embeddings.shape)}")
    if active_embeddings.dtype != torch.float16:
        raise ValueError(f"Expected FP16 projected embeddings, got {active_embeddings.dtype}")
    if len(row_offsets) != len(rows) + 1 or int(row_offsets[0]) != 0:
        raise ValueError("Invalid projected prompt row offsets")

    active_mask = attention_mask.bool()
    expected_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), active_mask.sum(dim=1).cumsum(dim=0)]
    )
    mask_coordinates = torch.nonzero(active_mask, as_tuple=False)
    if not torch.equal(row_offsets, expected_offsets):
        raise ValueError("Projected row offsets do not agree with the attention mask")
    if int(row_offsets[-1]) != len(active_embeddings):
        raise ValueError("Projected active embedding count does not agree with row offsets")
    if not torch.equal(active_positions, mask_coordinates[:, 1]):
        raise ValueError("Projected active positions do not agree with the attention mask")
    if not torch.equal(active_input_ids, input_ids[active_mask]):
        raise ValueError("Projected active token IDs do not agree with the dense token IDs")

    negative_mask = bundle["negative_prompt_attention_mask"].long()
    negative_ids = bundle["negative_prompt_input_ids"].long()
    negative_active = bundle["active_negative_prompt_embeds"]
    negative_offsets = bundle["negative_prompt_row_offsets"].long()
    negative_positions = bundle["active_negative_prompt_positions"].long()
    negative_active_ids = bundle["active_negative_prompt_input_ids"].long()
    if tuple(negative_mask.shape) != (1, 120) or tuple(negative_ids.shape) != (1, 120):
        raise ValueError("Expected negative prompt tensors shaped [1, 120]")
    if negative_active.ndim != 2 or negative_active.shape[1] != 4096:
        raise ValueError("Expected active negative embeddings shaped [tokens, 4096]")
    if negative_active.dtype != active_embeddings.dtype:
        raise ValueError("Positive and negative conditioning dtypes do not match")
    expected_negative_offsets = torch.tensor(
        [0, int(negative_mask.sum())],
        dtype=torch.int64,
    )
    negative_coordinates = torch.nonzero(negative_mask.bool(), as_tuple=False)
    if not torch.equal(negative_offsets, expected_negative_offsets):
        raise ValueError("Negative row offsets do not agree with the attention mask")
    if not torch.equal(negative_positions, negative_coordinates[:, 1]):
        raise ValueError("Negative active positions do not agree with the attention mask")
    if not torch.equal(negative_active_ids, negative_ids[negative_mask.bool()]):
        raise ValueError("Negative active token IDs do not agree with the dense token IDs")
    if len(negative_active) != int(negative_offsets[-1]):
        raise ValueError("Negative active embedding count does not agree with row offsets")
    if not torch.isfinite(active_embeddings).all() or not torch.isfinite(negative_active).all():
        raise ValueError("Projected conditioning contains non-finite embeddings")

    return bundle, file_sha256(path)


def reconstruct_prompt_conditioning(bundle, row_index, device):
    attention_mask = bundle["prompt_attention_mask"][row_index : row_index + 1].to(device)
    row_offsets = bundle["prompt_row_offsets"]
    start = int(row_offsets[row_index])
    end = int(row_offsets[row_index + 1])
    positions = bundle["active_prompt_positions"][start:end].long().to(device)
    active = bundle["active_prompt_embeds"][start:end].to(device)
    dense = torch.zeros(
        (1, attention_mask.shape[1], active.shape[1]),
        device=device,
        dtype=active.dtype,
    )
    dense[0, positions] = active
    return dense, attention_mask


def reconstruct_negative_conditioning(bundle, device):
    attention_mask = bundle["negative_prompt_attention_mask"].to(device)
    positions = bundle["active_negative_prompt_positions"].long().to(device)
    active = bundle["active_negative_prompt_embeds"].to(device)
    dense = torch.zeros(
        (1, attention_mask.shape[1], active.shape[1]),
        device=device,
        dtype=active.dtype,
    )
    dense[0, positions] = active
    return dense, attention_mask


def generate_images(args):
    from diffusers import DPMSolverMultistepScheduler, PixArtAlphaPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Image generation requires a CUDA GPU")
    if args.images_per_prompt < 1:
        raise ValueError("--images-per-prompt must be at least 1")

    rows = load_prompts().reset_index(drop=True)
    end_index = min(args.end_index, len(rows))
    if not 0 <= args.start_index < end_index:
        raise ValueError(
            f"Invalid prompt range [{args.start_index}, {args.end_index}); "
            f"the validation split has {len(rows)} prompts"
        )

    conditioning_bundle = None
    conditioning_sha256 = None
    if args.projected_text_embeddings is not None:
        conditioning_bundle, conditioning_sha256 = load_projected_conditioning(
            args.projected_text_embeddings,
            rows,
            args.model,
        )
        print(f"Projected conditioning: {args.projected_text_embeddings}")
        print(f"Projected SHA-256: {conditioning_sha256}")
        print(f"Projected active shape: {tuple(conditioning_bundle['active_prompt_embeds'].shape)}")

    image_dir = args.output_dir / "samples"
    metadata_dir = args.output_dir / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if conditioning_bundle is not None and any(image_dir.glob("*.png")):
        existing_metadata = sorted(metadata_dir.glob("prompts_*.jsonl"))
        if not existing_metadata:
            raise RuntimeError(
                "Projected output directory already has images but no metadata; use a new output directory"
            )
        for metadata_path in existing_metadata:
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("conditioning_mode") != "projected":
                        raise RuntimeError(
                            "Output directory contains non-projected images; use a separate projected directory"
                        )
                    if record.get("conditioning_sha256") != conditioning_sha256:
                        raise RuntimeError(
                            "Output directory contains images from a different projected artifact"
                        )

    pipeline_kwargs = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "revision": DEFAULT_MODEL_REVISION if args.model == DEFAULT_MODEL_ID else None,
        "token": False,
    }
    if conditioning_bundle is not None:
        pipeline_kwargs.update({"text_encoder": None, "tokenizer": None})

    pipe = PixArtAlphaPipeline.from_pretrained(args.model, **pipeline_kwargs)
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

    negative_prompt_embeds = None
    negative_prompt_attention_mask = None
    if conditioning_bundle is not None:
        generation_device = pipe._execution_device
        negative_prompt_embeds, negative_prompt_attention_mask = reconstruct_negative_conditioning(
            conditioning_bundle,
            generation_device,
        )

    records = []
    selected = rows.iloc[args.start_index:end_index]
    for row_index, row in tqdm(
        selected.iterrows(),
        total=len(selected),
        desc=f"prompts {args.start_index}:{end_index}",
    ):
        prompt = str(row["prompt"])
        prompt_id = int(row["prompt_id"])
        prompt_embeds = None
        prompt_attention_mask = None
        if conditioning_bundle is not None:
            prompt_embeds, prompt_attention_mask = reconstruct_prompt_conditioning(
                conditioning_bundle,
                int(row_index),
                pipe._execution_device,
            )

        for sample_index in range(args.images_per_prompt):
            image_name = f"{row_index:04d}_{sample_index:02d}.png"
            image_path = image_dir / image_name
            sample_seed = args.seed + row_index * args.images_per_prompt + sample_index

            if not image_path.exists():
                generator = torch.Generator(device="cpu").manual_seed(sample_seed)
                call_kwargs = {
                    "height": args.height,
                    "width": args.width,
                    "num_inference_steps": args.steps,
                    "guidance_scale": args.cfg_scale,
                    "generator": generator,
                    "num_images_per_prompt": 1,
                }
                if conditioning_bundle is None:
                    call_kwargs["prompt"] = prompt
                else:
                    call_kwargs.update(
                        {
                            "prompt": None,
                            "negative_prompt": None,
                            "prompt_embeds": prompt_embeds,
                            "prompt_attention_mask": prompt_attention_mask,
                            "negative_prompt_embeds": negative_prompt_embeds,
                            "negative_prompt_attention_mask": negative_prompt_attention_mask,
                        }
                    )
                image = pipe(**call_kwargs).images[0]
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
                    "model_revision": DEFAULT_MODEL_REVISION if args.model == DEFAULT_MODEL_ID else None,
                    "height": args.height,
                    "width": args.width,
                    "steps": args.steps,
                    "cfg_scale": args.cfg_scale,
                    "conditioning_mode": (
                        "projected" if conditioning_bundle is not None else "native_t5"
                    ),
                    "conditioning_artifact": (
                        str(args.projected_text_embeddings.resolve())
                        if args.projected_text_embeddings is not None
                        else None
                    ),
                    "conditioning_sha256": conditioning_sha256,
                    "conditioning_source_model": (
                        conditioning_bundle["metadata"].get("source_model")
                        if conditioning_bundle is not None
                        else None
                    ),
                    "conditioning_projector": (
                        conditioning_bundle["metadata"].get("projector_format")
                        if conditioning_bundle is not None
                        else None
                    ),
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

    mean_score = scores.mean().item()
    conditioning_modes = sorted(
        {record.get("conditioning_mode", "native_t5") for record in scored_records}
    )
    conditioning_hashes = sorted(
        {
            record["conditioning_sha256"]
            for record in scored_records
            if record.get("conditioning_sha256") is not None
        }
    )
    summary = {
        "metric": "T2I-CompBench non-spatial CLIP ViT-B/32 cosine",
        "num_images": len(scored_records),
        "num_prompts": len({record["row_index"] for record in scored_records}),
        "images_per_prompt": sorted(set(prompt_counts.values())),
        "complete_validation_split": len(prompt_counts) == 300,
        "conditioning_modes": conditioning_modes,
        "conditioning_sha256": conditioning_hashes,
        "mean": mean_score,
        "std": scores.std(unbiased=False).item(),
        "paper_pixart_alpha": PAPER_SCORE,
        "difference_from_paper": mean_score - PAPER_SCORE,
        "native_one_image_baseline": NATIVE_ONE_IMAGE_BASELINE_SCORE,
        "difference_from_native_baseline": mean_score - NATIVE_ONE_IMAGE_BASELINE_SCORE,
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
