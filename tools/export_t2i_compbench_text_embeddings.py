"""Export the exact text conditioning used by the lightweight baseline.

The output is a portable torch bundle containing the positive T5 hidden
states for the 300 T2I-CompBench non-spatial prompts, their masks and token
IDs, and the empty-prompt conditioning used by Diffusers for CFG.  The dense
prompt embedding shape is [number of prompts, 120, 4096].
"""

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.eval_t2i_compbench_nonspatial import (  # noqa: E402
    DATASET_FILE,
    DATASET_ID,
    DATASET_REVISION,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_prompts,
)


MAX_SEQUENCE_LENGTH = 120
EXPECTED_HIDDEN_SIZE = 4096


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export native PixArt T5 conditioning for the CompBench baseline prompts"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision; the pinned baseline revision is used for the default model",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(pipe, prompts):
    return pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


def encode(pipe, prompts, device):
    with torch.inference_mode():
        prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
            prompt=prompts,
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            device=device,
            clean_caption=True,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
    return prompt_embeds, prompt_attention_mask


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    try:
        import bs4  # noqa: F401
        import ftfy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Exact PixArt caption cleaning requires beautifulsoup4 and ftfy") from exc

    from diffusers import PixArtAlphaPipeline, __version__ as diffusers_version
    from transformers import __version__ as transformers_version

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    rows = load_prompts().reset_index(drop=True)
    end_index = min(args.end_index, len(rows))
    if not 0 <= args.start_index < end_index:
        raise ValueError(
            f"Invalid prompt range [{args.start_index}, {args.end_index}); "
            f"the validation split has {len(rows)} prompts"
        )
    selected = rows.iloc[args.start_index:end_index]
    prompts = [str(prompt) for prompt in selected["prompt"].tolist()]
    prompt_ids = [int(prompt_id) for prompt_id in selected["prompt_id"].tolist()]
    row_indices = list(range(args.start_index, end_index))

    revision = args.revision
    if revision is None and args.model == DEFAULT_MODEL_ID:
        revision = DEFAULT_MODEL_REVISION

    load_kwargs = {
        "transformer": None,
        "vae": None,
        "torch_dtype": torch.float16,
        "use_safetensors": True,
        "token": False,
    }
    if revision is not None:
        load_kwargs["revision"] = revision

    print(f"Loading tokenizer and T5 encoder from {args.model} at {revision or 'default revision'}")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model, **load_kwargs)
    if pipe.text_encoder is None or pipe.tokenizer is None:
        raise RuntimeError("The selected pipeline does not contain a tokenizer and text encoder")
    hidden_size = int(pipe.text_encoder.config.d_model)
    if hidden_size != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(f"Expected T5 hidden size {EXPECTED_HIDDEN_SIZE}, found {hidden_size}")
    pipe.text_encoder.eval().to(device)

    # This is the same private preprocessing method called by the pipeline's
    # public generation path when clean_caption=True.
    cleaned_prompts = pipe._text_preprocessing(prompts, clean_caption=True)
    prompt_tokens = tokenize(pipe, cleaned_prompts)

    embedding_batches = []
    mask_batches = []
    for offset in tqdm(range(0, len(prompts), args.batch_size), desc="Encoding prompts"):
        batch = prompts[offset : offset + args.batch_size]
        batch_embeds, batch_mask = encode(pipe, batch, device)
        expected_mask = prompt_tokens.attention_mask[offset : offset + len(batch)]
        if not torch.equal(batch_mask.cpu(), expected_mask):
            raise RuntimeError(f"Attention-mask mismatch in batch starting at prompt {offset}")
        embedding_batches.append(batch_embeds.to(device="cpu", dtype=torch.float16).contiguous())
        mask_batches.append(batch_mask.to(device="cpu", dtype=torch.int64).contiguous())

    prompt_embeds = torch.cat(embedding_batches, dim=0)
    prompt_attention_mask = torch.cat(mask_batches, dim=0)

    negative_prompt = ""
    cleaned_negative_prompt = pipe._text_preprocessing([negative_prompt], clean_caption=True)
    negative_tokens = tokenize(pipe, cleaned_negative_prompt)
    negative_prompt_embeds, negative_prompt_attention_mask = encode(pipe, [negative_prompt], device)
    negative_prompt_embeds = negative_prompt_embeds.to(
        device="cpu", dtype=torch.float16
    ).contiguous()
    negative_prompt_attention_mask = negative_prompt_attention_mask.to(
        device="cpu", dtype=torch.int64
    ).contiguous()

    expected_shape = (len(prompts), MAX_SEQUENCE_LENGTH, EXPECTED_HIDDEN_SIZE)
    if tuple(prompt_embeds.shape) != expected_shape:
        raise RuntimeError(f"Expected prompt embeddings shaped {expected_shape}, got {tuple(prompt_embeds.shape)}")
    if tuple(negative_prompt_embeds.shape) != (1, MAX_SEQUENCE_LENGTH, EXPECTED_HIDDEN_SIZE):
        raise RuntimeError(f"Unexpected negative embedding shape: {tuple(negative_prompt_embeds.shape)}")
    if not torch.isfinite(prompt_embeds).all() or not torch.isfinite(negative_prompt_embeds).all():
        raise RuntimeError("Export contains non-finite embeddings")
    if not torch.equal(negative_prompt_attention_mask, negative_tokens.attention_mask):
        raise RuntimeError("Negative-prompt attention-mask mismatch")

    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)

    bundle = {
        "format": "pixart_t2i_compbench_native_conditioning_v1",
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
        "prompt_input_ids": prompt_tokens.input_ids.to(dtype=torch.int64).contiguous(),
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_attention_mask": negative_prompt_attention_mask,
        "negative_prompt_input_ids": negative_tokens.input_ids.to(dtype=torch.int64).contiguous(),
        "prompts": prompts,
        "cleaned_prompts": cleaned_prompts,
        "prompt_sha256": [sha256_text(prompt) for prompt in prompts],
        "cleaned_prompt_sha256": [sha256_text(prompt) for prompt in cleaned_prompts],
        "row_indices": torch.tensor(row_indices, dtype=torch.int64),
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.int64),
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_type": "native_positive_and_cfg_negative_text_conditioning",
            "conditioning_boundary": "T5EncoderModel last_hidden_state before PixArt caption_projection",
            "model": args.model,
            "model_revision": revision,
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_file": DATASET_FILE,
            "start_index": args.start_index,
            "end_index": end_index,
            "complete_validation_split": args.start_index == 0 and end_index == len(rows),
            "num_prompts": len(prompts),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "hidden_size": EXPECTED_HIDDEN_SIZE,
            "embedding_dtype": str(prompt_embeds.dtype),
            "clean_caption": True,
            "preprocessing": "PixArtAlphaPipeline._text_preprocessing; clean_caption applied twice",
            "negative_prompt": negative_prompt,
            "diffusers_version": diffusers_version,
            "transformers_version": transformers_version,
            "torch_version": str(torch.__version__),
            "extraction_device": str(device),
            "gpu_name": gpu_name,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    file_digest = sha256_file(args.output)
    checksum_path = Path(f"{args.output}.sha256")
    checksum_path.write_text(f"{file_digest}  {args.output.name}\n", encoding="utf-8")

    size_mib = args.output.stat().st_size / (1024**2)
    print(f"Saved: {args.output}")
    print(f"Shape: {tuple(prompt_embeds.shape)}")
    print(f"Active tokens: {int(prompt_attention_mask.sum())}")
    print(f"Size: {size_mib:.1f} MiB")
    print(f"SHA-256: {file_digest}")
    print(f"Checksum: {checksum_path}")


if __name__ == "__main__":
    main()
