# %% [markdown]
# # VLM embeddings for PixArt-alpha
#
# This Kaggle notebook mirrors the earlier SANA projection workflow for the
# 300 T2I-CompBench non-spatial PixArt baseline prompts. It keeps occurrence-
# aware SEAM character-span alignment and polynomial kernel ridge regression,
# but uses raw LLaVA language-model input embeddings as the source and native
# PixArt T5 `last_hidden_state` token embeddings as the target.

# %%
!pip install -U transformers accelerate sentencepiece protobuf nltk safetensors huggingface_hub
!pip install qwen-vl-utils

# %% [markdown]
# ## Clone the PixArt fork

# %%
from pathlib import Path

REPO = Path("/kaggle/working/PixArt-alpha")

if REPO.exists():
    !git -C {REPO} pull origin master
else:
    !git clone --depth 1 --branch master https://github.com/aveen007/PixArt-alpha.git {REPO}

%cd /kaggle/working/PixArt-alpha

# %% [markdown]
# ## Locate and validate the downloaded native PixArt token bundle

# %%
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from IPython.display import FileLink, display
from tqdm.auto import tqdm

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def find_one(filename):
    matches = sorted(Path("/kaggle/input").rglob(filename))
    assert matches, f"{filename} was not found under /kaggle/input"
    assert len(matches) == 1, f"Found multiple copies of {filename}: {matches}"
    return matches[0]


NATIVE_PATH = find_one("native_t5_active_tokens_fp16.pt")

native = torch.load(
    NATIVE_PATH,
    map_location="cpu",
    weights_only=True,
)

required_keys = {
    "format",
    "active_prompt_embeds",
    "prompt_row_offsets",
    "active_prompt_positions",
    "active_prompt_input_ids",
    "prompt_attention_mask",
    "prompt_input_ids",
    "active_negative_prompt_embeds",
    "negative_prompt_attention_mask",
    "negative_prompt_input_ids",
    "prompts",
    "cleaned_prompts",
    "prompt_sha256",
    "cleaned_prompt_sha256",
    "row_indices",
    "prompt_ids",
    "metadata",
}

missing = required_keys - set(native)
assert not missing, f"Native bundle is missing keys: {sorted(missing)}"
assert native["format"] == "pixart_t2i_compbench_active_conditioning_v2"

prompts = list(native["prompts"])
conditioned_texts = list(native["cleaned_prompts"])
target_mask = native["prompt_attention_mask"].bool()
target_ids = native["prompt_input_ids"].long()
target_active = native["active_prompt_embeds"].contiguous()
target_row_offsets = native["prompt_row_offsets"].long()
target_active_positions = native["active_prompt_positions"].long()
target_active_ids = native["active_prompt_input_ids"].long()

assert len(prompts) == 300
assert len(conditioned_texts) == len(prompts)
assert target_mask.shape == (len(prompts), 120)
assert target_ids.shape == target_mask.shape
assert target_active.shape[1] == 4096
assert int(target_row_offsets[-1]) == len(target_active)
assert len(target_row_offsets) == len(prompts) + 1

mask_coordinates = torch.nonzero(target_mask, as_tuple=False)
assert torch.equal(mask_coordinates[:, 1], target_active_positions)
assert torch.equal(target_ids[target_mask], target_active_ids)

expected_prompt_hashes = [
    hashlib.sha256(text.encode("utf-8")).hexdigest()
    for text in prompts
]
expected_cleaned_hashes = [
    hashlib.sha256(text.encode("utf-8")).hexdigest()
    for text in conditioned_texts
]
assert expected_prompt_hashes == native["prompt_sha256"]
assert expected_cleaned_hashes == native["cleaned_prompt_sha256"]

print("Native file:", NATIVE_PATH)
print("Native format:", native["format"])
print("Prompts:", len(prompts))
print("Active target tokens:", len(target_active))
print("Target active shape:", tuple(target_active.shape))
print("Logical shape:", native["metadata"]["logical_prompt_embedding_shape"])

# %% [markdown]
# ## Reconstruct exact T5 offsets and special-token flags
#
# The compact file contains exact IDs and masks but intentionally omits character
# offsets. A fast T5 tokenizer reconstructs them here, and the cell refuses to
# continue unless every ID and mask matches the exported baseline artifact.

# %%
from transformers import AutoTokenizer

target_model_id = native["metadata"]["model"]
target_revision = native["metadata"]["model_revision"]

target_tokenizer = AutoTokenizer.from_pretrained(
    target_model_id,
    subfolder="tokenizer",
    revision=target_revision,
    use_fast=True,
    token=False,
)

assert target_tokenizer.is_fast, "A fast T5 tokenizer is required for offsets"

target_encoded = target_tokenizer(
    conditioned_texts,
    max_length=120,
    padding="max_length",
    truncation=True,
    add_special_tokens=True,
    return_attention_mask=True,
    return_offsets_mapping=True,
    return_special_tokens_mask=True,
    return_tensors="pt",
)

assert torch.equal(target_encoded["input_ids"], target_ids)
assert torch.equal(target_encoded["attention_mask"].bool(), target_mask)

target_offsets = target_encoded["offset_mapping"].long()
target_special_mask = target_encoded["special_tokens_mask"].bool()

active_special_count = int(target_special_mask[target_mask].sum())
print("Exact target tokenization verified")
print("Active special tokens:", active_special_count)

# %% [markdown]
# ## Load only LLaVA's raw language-model input embedding table
#
# This does not load the vision tower and does not run a LLaVA forward pass.
# Only the tokenizer, safetensors index, and shard containing `embed_tokens`
# are downloaded.

# %%
from huggingface_hub import hf_hub_download
from safetensors import safe_open

SOURCE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

source_tokenizer = AutoTokenizer.from_pretrained(
    SOURCE_MODEL_ID,
    use_fast=True,
    token=False,
)
assert source_tokenizer.is_fast
source_tokenizer.padding_side = "right"

if source_tokenizer.pad_token_id is None:
    source_tokenizer.pad_token = (
        source_tokenizer.eos_token
        or source_tokenizer.unk_token
    )

index_path = hf_hub_download(
    repo_id=SOURCE_MODEL_ID,
    filename="model.safetensors.index.json",
    token=False,
)

with open(index_path, "r", encoding="utf-8") as handle:
    weight_map = json.load(handle)["weight_map"]

embedding_keys = sorted(
    key
    for key in weight_map
    if key.endswith("embed_tokens.weight")
)

preferred_keys = [
    key for key in embedding_keys
    if "language_model" in key
]

if len(preferred_keys) == 1:
    embedding_key = preferred_keys[0]
elif len(embedding_keys) == 1:
    embedding_key = embedding_keys[0]
else:
    raise RuntimeError(f"Could not choose the LLaVA embedding key: {embedding_keys}")

embedding_shard_name = weight_map[embedding_key]
embedding_shard_path = hf_hub_download(
    repo_id=SOURCE_MODEL_ID,
    filename=embedding_shard_name,
    token=False,
)

with safe_open(embedding_shard_path, framework="pt", device="cpu") as handle:
    source_embedding_table = handle.get_tensor(embedding_key).clone()

assert source_embedding_table.ndim == 2
assert source_embedding_table.shape[1] == 4096
assert source_embedding_table.shape[0] >= len(source_tokenizer)

print("Source model:", SOURCE_MODEL_ID)
print("Embedding key:", embedding_key)
print("Embedding shard:", embedding_shard_name)
print("Embedding table:", tuple(source_embedding_table.shape), source_embedding_table.dtype)

# %% [markdown]
# ## SEAM alignment: occurrence-aware character overlap
#
# Both tokenizers see the exact same PixArt-cleaned string. Each active regular
# T5 token receives the overlap-weighted mean of raw LLaVA input embeddings.
# Active special tokens are paired deterministically in occurrence order.
# Nearest-span fallbacks are recorded explicitly.

# %%
SOURCE_BATCH_SIZE = 64


@torch.inference_mode()
def extract_seam_pairs():
    x_parts = []
    y_parts = []
    prompt_index_parts = []
    target_position_parts = []
    fallback_parts = []

    alignment_row_offsets = [0]
    alignment_source_positions = []
    alignment_source_weights = []

    for batch_start in tqdm(
        range(0, len(conditioned_texts), SOURCE_BATCH_SIZE),
        desc="SEAM-aligning LLaVA lookup embeddings",
    ):
        batch_end = min(batch_start + SOURCE_BATCH_SIZE, len(conditioned_texts))
        batch_texts = conditioned_texts[batch_start:batch_end]

        encoded = source_tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        source_ids = encoded["input_ids"]
        source_attention = encoded["attention_mask"].bool()
        source_offsets = encoded["offset_mapping"].long()
        source_special = encoded["special_tokens_mask"].bool()
        source_lookup = source_embedding_table[source_ids]

        for local_index, _ in enumerate(batch_texts):
            prompt_index = batch_start + local_index
            target_positions = torch.nonzero(
                target_mask[prompt_index],
                as_tuple=False,
            ).flatten()

            active_start = int(target_row_offsets[prompt_index])
            active_end = int(target_row_offsets[prompt_index + 1])
            targets = target_active[active_start:active_end]

            assert len(target_positions) == len(targets)
            assert torch.equal(
                target_positions,
                target_active_positions[active_start:active_end],
            )

            target_is_special = target_special_mask[
                prompt_index,
                target_positions,
            ]

            aligned = torch.empty(
                len(target_positions),
                source_embedding_table.shape[1],
                dtype=torch.float32,
            )
            local_fallback = torch.zeros(len(target_positions), dtype=torch.bool)
            local_source_positions = [None] * len(target_positions)
            local_source_weights = [None] * len(target_positions)

            row_attention = source_attention[local_index]
            row_special = source_special[local_index]
            row_offsets = source_offsets[local_index]
            row_lookup = source_lookup[local_index].float()

            source_regular_indices = torch.nonzero(
                row_attention
                & ~row_special
                & (row_offsets[:, 1] > row_offsets[:, 0]),
                as_tuple=False,
            ).flatten()

            regular_target_local = torch.nonzero(
                ~target_is_special,
                as_tuple=False,
            ).flatten()

            if len(regular_target_local):
                assert len(source_regular_indices), "No regular LLaVA tokens found"

                regular_target_positions = target_positions[regular_target_local]
                target_spans = target_offsets[
                    prompt_index,
                    regular_target_positions,
                ]
                source_spans = row_offsets[source_regular_indices]

                overlap_start = torch.maximum(
                    target_spans[:, None, 0],
                    source_spans[None, :, 0],
                )
                overlap_end = torch.minimum(
                    target_spans[:, None, 1],
                    source_spans[None, :, 1],
                )
                overlap = (overlap_end - overlap_start).clamp_min(0).float()

                missing = overlap.sum(dim=1) == 0
                if missing.any():
                    target_centers = target_spans.float().mean(dim=1)
                    source_centers = source_spans.float().mean(dim=1)

                    for missing_row in torch.nonzero(missing, as_tuple=False).flatten():
                        nearest = torch.argmin(
                            torch.abs(source_centers - target_centers[missing_row])
                        )
                        overlap[missing_row, nearest] = 1.0
                        local_fallback[regular_target_local[missing_row]] = True

                weights = overlap / overlap.sum(dim=1, keepdim=True).clamp_min(1)
                aligned[regular_target_local] = weights @ row_lookup[source_regular_indices]

                for regular_row, target_local in enumerate(regular_target_local.tolist()):
                    used = torch.nonzero(
                        weights[regular_row] > 0,
                        as_tuple=False,
                    ).flatten()
                    local_source_positions[target_local] = source_regular_indices[used].tolist()
                    local_source_weights[target_local] = weights[regular_row, used].tolist()

            special_target_local = torch.nonzero(
                target_is_special,
                as_tuple=False,
            ).flatten()
            source_special_indices = torch.nonzero(
                row_attention & row_special,
                as_tuple=False,
            ).flatten()

            if len(special_target_local):
                assert len(source_special_indices), "LLaVA produced no active special token"

                for special_order, target_local in enumerate(special_target_local.tolist()):
                    source_position = source_special_indices[
                        min(special_order, len(source_special_indices) - 1)
                    ]
                    aligned[target_local] = row_lookup[source_position]
                    local_source_positions[target_local] = [int(source_position)]
                    local_source_weights[target_local] = [1.0]

            for source_positions, source_weights in zip(
                local_source_positions,
                local_source_weights,
            ):
                assert source_positions is not None
                assert source_weights is not None
                alignment_source_positions.extend(source_positions)
                alignment_source_weights.extend(source_weights)
                alignment_row_offsets.append(len(alignment_source_positions))

            x_parts.append(aligned.to(torch.float16))
            y_parts.append(targets.to(torch.float16))
            prompt_index_parts.extend([prompt_index] * len(target_positions))
            target_position_parts.extend(target_positions.tolist())
            fallback_parts.append(local_fallback)

    X = torch.cat(x_parts).contiguous()
    Y = torch.cat(y_parts).contiguous()
    pair_prompt_indices = torch.tensor(prompt_index_parts, dtype=torch.int32)
    pair_target_positions = torch.tensor(target_position_parts, dtype=torch.int16)
    fallback_mask = torch.cat(fallback_parts).contiguous()

    return {
        "X": X,
        "Y": Y,
        "prompt_indices": pair_prompt_indices,
        "target_positions": pair_target_positions,
        "alignment_fallback_mask": fallback_mask,
        "alignment_row_offsets": torch.tensor(alignment_row_offsets, dtype=torch.int64),
        "alignment_source_positions": torch.tensor(
            alignment_source_positions,
            dtype=torch.int32,
        ),
        "alignment_source_weights": torch.tensor(
            alignment_source_weights,
            dtype=torch.float32,
        ),
    }


alignment_start = time.perf_counter()
pairs = extract_seam_pairs()
alignment_seconds = time.perf_counter() - alignment_start

X_all = pairs["X"]
Y_all = pairs["Y"]
pair_prompt_indices = pairs["prompt_indices"]
pair_target_positions = pairs["target_positions"]

assert len(X_all) == len(Y_all) == int(target_mask.sum())
assert len(pairs["alignment_row_offsets"]) == len(X_all) + 1
assert int(pairs["alignment_row_offsets"][-1]) == len(
    pairs["alignment_source_positions"]
)
assert torch.equal(pair_prompt_indices.long(), mask_coordinates[:, 0])
assert torch.equal(pair_target_positions.long(), mask_coordinates[:, 1])

PAIR_PATH = Path("/kaggle/working/pixart_llava_compbench_seam_pairs.pt")
pair_bundle = {
    "format": "pixart_llava_compbench_seam_pairs_v1",
    **pairs,
    "target_input_ids": target_ids,
    "target_attention_mask": target_mask,
    "active_target_input_ids": target_active_ids,
    "prompts": prompts,
    "conditioned_texts": conditioned_texts,
    "prompt_sha256": native["prompt_sha256"],
    "source_model": SOURCE_MODEL_ID,
    "source_representation": "raw_language_model_input_embedding_lookup",
    "target_model": target_model_id,
    "target_representation": "PixArt_T5_last_hidden_state_before_caption_projection",
    "alignment": "SEAM_character_overlap_weighted",
    "alignment_seconds": alignment_seconds,
}
torch.save(pair_bundle, PAIR_PATH)

print("Aligned pairs:", tuple(X_all.shape), tuple(Y_all.shape))
print("Fallbacks:", int(pairs["alignment_fallback_mask"].sum()))
print("Alignment seconds:", round(alignment_seconds, 2))
print("Saved:", PAIR_PATH)
display(FileLink(str(PAIR_PATH)))

# %% [markdown]
# ## Keep one deterministic occurrence per unique target T5 token ID

# %%
all_pair_token_ids = target_ids[
    pair_prompt_indices.long(),
    pair_target_positions.long(),
].long()

assert torch.equal(all_pair_token_ids, target_active_ids)

first_row_by_token_id = {}
for occurrence_row, token_id in enumerate(all_pair_token_ids.tolist()):
    if token_id not in first_row_by_token_id:
        first_row_by_token_id[token_id] = occurrence_row

fit_rows = torch.tensor(
    list(first_row_by_token_id.values()),
    dtype=torch.long,
)

heldout_mask = torch.ones(len(X_all), dtype=torch.bool)
heldout_mask[fit_rows] = False
heldout_rows = torch.nonzero(heldout_mask, as_tuple=False).flatten()

X_cpu = X_all.index_select(0, fit_rows).contiguous()
Y_cpu = Y_all.index_select(0, fit_rows).contiguous()
fit_token_ids = all_pair_token_ids.index_select(0, fit_rows).contiguous()

assert len(torch.unique(fit_token_ids)) == len(fit_token_ids)

del source_embedding_table
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

kernel_gib = len(X_cpu) * len(X_cpu) * 4 / 1024**3
print("All aligned occurrences:", len(X_all))
print("Unique fitting token IDs:", len(X_cpu))
print("Repeated held-out occurrences:", len(heldout_rows))
print("Expected float32 kernel:", round(kernel_gib, 3), "GiB")

# %% [markdown]
# ## Prepare direction embeddings and Cholesky whitening

# %%
fit_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_raw = X_cpu.float().to(fit_device)
Y_raw = Y_cpu.float().to(fit_device)


def normalize_rows(tensor, eps=1e-12):
    return tensor / tensor.norm(dim=1, keepdim=True).clamp_min(eps)


def fit_cholesky_whitener(tensor, eps=1e-3):
    tensor = tensor.float()
    mean = tensor.mean(dim=0, keepdim=True)
    centered = tensor - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    covariance = covariance + eps * torch.eye(
        covariance.shape[0],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    cholesky = torch.linalg.cholesky(covariance)
    white = torch.linalg.solve_triangular(
        cholesky,
        centered.T,
        upper=False,
    ).T
    return white, {"mean": mean, "L": cholesky}


def invert_cholesky_whitener(white, state):
    return white @ state["L"].T + state["mean"]


X_norm = normalize_rows(X_raw)
X_norm_mean = X_norm.mean(dim=0, keepdim=True)
X_norm = normalize_rows(X_norm - X_norm_mean)

Y_runtime = normalize_rows(Y_raw)
Y_norm, Y_chol_state = fit_cholesky_whitener(Y_runtime, eps=1e-3)

print("Fit device:", fit_device)
print("X:", tuple(X_raw.shape), "->", tuple(X_norm.shape))
print("Y:", tuple(Y_raw.shape), "->", tuple(Y_norm.shape))

# %% [markdown]
# ## Metrics

# %%
def projection_metrics(projected, target, batch_size=256):
    projected = normalize_rows(projected)
    target = normalize_rows(target)
    assert projected.shape == target.shape

    row_count = len(projected)
    correct_values = F.cosine_similarity(projected, target, dim=1)
    wrong_sum = 0.0
    wrong_count = 0
    top1_correct = 0

    for start in range(0, row_count, batch_size):
        end = min(start + batch_size, row_count)
        similarities = projected[start:end] @ target.T
        local_rows = torch.arange(end - start, device=similarities.device)
        correct_cols = torch.arange(start, end, device=similarities.device)
        diagonal = similarities[local_rows, correct_cols]
        wrong_sum += float(similarities.sum() - diagonal.sum())
        wrong_count += similarities.numel() - len(diagonal)
        top1_correct += int((similarities.argmax(dim=1) == correct_cols).sum())

    wrong_mean = wrong_sum / wrong_count if wrong_count else float("nan")
    l2 = torch.norm(projected - target, dim=1)
    metrics = {
        "correct_cosine": float(correct_values.mean()),
        "wrong_cosine": wrong_mean,
        "separation_gap": float(correct_values.mean()) - wrong_mean,
        "top1": top1_correct / row_count,
        "mse": float(F.mse_loss(projected, target)),
        "mean_l2": float(l2.mean()),
        "minimum_cosine": float(correct_values.min()),
        "maximum_cosine": float(correct_values.max()),
    }

    for key, value in metrics.items():
        print(f"{key:20s}: {value:.6f}")
    return metrics


def norm_metrics(projected, target):
    predicted_norm = projected.float().norm(dim=1)
    target_norm = target.float().norm(dim=1)
    correlation = float("nan")
    if len(predicted_norm) > 1:
        correlation = float(torch.corrcoef(torch.stack([predicted_norm, target_norm]))[0, 1])
    return {
        "predicted_mean": float(predicted_norm.mean()),
        "target_mean": float(target_norm.mean()),
        "mae": float((predicted_norm - target_norm).abs().mean()),
        "correlation": correlation,
    }

# %% [markdown]
# ## Polynomial KRR direction head

# %%
def polynomial_kernel(X1, X2, sigma=0.5003, degree=2, coef0=0.2):
    return (sigma * (X1 @ X2.T) + coef0) ** degree


sigma = 0.5003
degree = 2
coef0 = 0.2
requested_lambda = 1e-5

fit_start = time.perf_counter()
K = polynomial_kernel(X_norm, X_norm, sigma=sigma, degree=degree, coef0=coef0)
K = (K + K.T) * 0.5
identity = torch.eye(len(K), device=fit_device, dtype=K.dtype)

kernel_cholesky = None
lambda_reg = None
for multiplier in [1, 3, 10, 30, 100, 300, 1000]:
    candidate_lambda = requested_lambda * multiplier
    candidate, info = torch.linalg.cholesky_ex(
        K + candidate_lambda * identity
    )
    if int(info.max()) == 0:
        kernel_cholesky = candidate
        lambda_reg = candidate_lambda
        break

assert kernel_cholesky is not None
direction_coefficients = torch.cholesky_solve(Y_norm, kernel_cholesky)
fit_seconds = time.perf_counter() - fit_start

print("Kernel:", tuple(K.shape))
print("Direction coefficients:", tuple(direction_coefficients.shape))
print("Lambda:", lambda_reg)
print("Direction fit seconds:", round(fit_seconds, 2))

# %% [markdown]
# ## Project directions and evaluate repeated-token occurrences

# %%
def transform_source(source):
    source = normalize_rows(source)
    return normalize_rows(source - X_norm_mean)


@torch.inference_mode()
def reconstruct_direction(kernel_rows):
    predicted_white = kernel_rows @ direction_coefficients
    predicted = invert_cholesky_whitener(predicted_white, Y_chol_state)
    return normalize_rows(predicted)


@torch.inference_mode()
def project_directions(source_rows, batch_size=256):
    output = []
    for start in tqdm(range(0, len(source_rows), batch_size), desc="Projecting directions"):
        end = min(start + batch_size, len(source_rows))
        source = source_rows[start:end].float().to(fit_device)
        source = transform_source(source)
        kernel_rows = polynomial_kernel(
            source,
            X_norm,
            sigma=sigma,
            degree=degree,
            coef0=coef0,
        )
        output.append(reconstruct_direction(kernel_rows).cpu())
    return torch.cat(output).contiguous()


fit_direction = reconstruct_direction(K)
print("TRAINING UNIQUE TOKENS / DIRECTION")
train_direction_metrics = projection_metrics(fit_direction, Y_runtime)

all_direction_cpu = project_directions(X_all)

if len(heldout_rows):
    heldout_direction = all_direction_cpu.index_select(0, heldout_rows).to(fit_device)
    heldout_target_direction = normalize_rows(
        Y_all.index_select(0, heldout_rows).float().to(fit_device)
    )
    print("\nREPEATED OCCURRENCES / DIRECTION")
    heldout_direction_metrics = projection_metrics(
        heldout_direction,
        heldout_target_direction,
    )
else:
    heldout_direction_metrics = None

# %% [markdown]
# ## Separate log-L2-norm KRR head

# %%
target_norm = Y_raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
target_log_norm = target_norm.log()
log_norm_mean = target_log_norm.mean(dim=0, keepdim=True)
log_norm_std = target_log_norm.std(dim=0, keepdim=True).clamp_min(1e-6)
target_log_norm_z = (target_log_norm - log_norm_mean) / log_norm_std
norm_min = float(target_norm.min())
norm_max = float(target_norm.max())

norm_coefficients = torch.cholesky_solve(
    target_log_norm_z,
    kernel_cholesky,
)


@torch.inference_mode()
def predict_norms(source_rows, batch_size=256):
    output = []
    for start in tqdm(range(0, len(source_rows), batch_size), desc="Projecting token norms"):
        end = min(start + batch_size, len(source_rows))
        source = transform_source(source_rows[start:end].float().to(fit_device))
        kernel_rows = polynomial_kernel(
            source,
            X_norm,
            sigma=sigma,
            degree=degree,
            coef0=coef0,
        )
        predicted_z = kernel_rows @ norm_coefficients
        predicted_log = predicted_z * log_norm_std + log_norm_mean
        output.append(predicted_log.exp().clamp(norm_min, norm_max).cpu())
    return torch.cat(output).contiguous()


all_predicted_norms = predict_norms(X_all)
projected_active = normalize_rows(all_direction_cpu.float()) * all_predicted_norms

fit_projected = projected_active.index_select(0, fit_rows)
train_norm_metrics = norm_metrics(fit_projected, Y_all.index_select(0, fit_rows))

if len(heldout_rows):
    heldout_norm_metrics = norm_metrics(
        projected_active.index_select(0, heldout_rows),
        Y_all.index_select(0, heldout_rows),
    )
else:
    heldout_norm_metrics = None

print("Training norm metrics:", train_norm_metrics)
print("Repeated-occurrence norm metrics:", heldout_norm_metrics)
print("Projected active shape:", tuple(projected_active.shape))

# %% [markdown]
# ## Save the projector and projected PixArt conditioning
#
# The projected file keeps the native bundle's exact prompt order, token IDs,
# masks, row offsets, hashes, and native empty-prompt CFG conditioning. Only
# `active_prompt_embeds` is replaced.

# %%
PROJECTOR_PATH = Path("/kaggle/working/pixart_llava_compbench_projector.pt")
PROJECTED_PATH = Path(
    "/kaggle/working/pixart_llava_compbench_projected_active_fp16.pt"
)

projector_bundle = {
    "format": "pixart_llava_polynomial_krr_projector_v1",
    "source_model": SOURCE_MODEL_ID,
    "source_representation": "raw_language_model_input_embedding_lookup",
    "source_embedding_key": embedding_key,
    "target_model": target_model_id,
    "target_model_revision": target_revision,
    "target_representation": "PixArt_T5_last_hidden_state_before_caption_projection",
    "alignment": "SEAM_character_overlap_weighted_occurrence_aware",
    "alignment_fallback_count": int(pairs["alignment_fallback_mask"].sum()),
    "calculation_prompt_count": len(prompts),
    "aligned_occurrence_count": len(X_all),
    "training_pair_policy": "first_occurrence_per_unique_target_token_id",
    "training_pair_count": len(X_cpu),
    "training_rows": fit_rows,
    "training_token_ids": fit_token_ids,
    "training_prompt_indices": pair_prompt_indices.index_select(0, fit_rows),
    "training_target_positions": pair_target_positions.index_select(0, fit_rows),
    "X_norm_mean": X_norm_mean.cpu(),
    "train_features": X_norm.cpu(),
    "kernel": {
        "kind": "polynomial",
        "sigma": sigma,
        "degree": degree,
        "coef0": coef0,
        "lambda_reg": lambda_reg,
    },
    "direction_coefficients": direction_coefficients.cpu(),
    "Y_mean": Y_chol_state["mean"].cpu(),
    "Y_cholesky": Y_chol_state["L"].cpu(),
    "Y_cholesky_epsilon": 1e-3,
    "norm_coefficients": norm_coefficients.cpu(),
    "Y_log_norm_mean": log_norm_mean.cpu(),
    "Y_log_norm_std": log_norm_std.cpu(),
    "Y_norm_min": norm_min,
    "Y_norm_max": norm_max,
    "training_direction_metrics": train_direction_metrics,
    "repeated_occurrence_direction_metrics": heldout_direction_metrics,
    "training_norm_metrics": train_norm_metrics,
    "repeated_occurrence_norm_metrics": heldout_norm_metrics,
    "fit_seconds": fit_seconds,
    "evaluation_note": (
        "Sanity check on the same 300-prompt vocabulary; repeated occurrences "
        "are not an independent vocabulary split."
    ),
}

projected_bundle = dict(native)
projected_bundle["format"] = "pixart_t2i_compbench_projected_active_conditioning_v1"
projected_bundle["active_prompt_embeds"] = projected_active.to(torch.float16)
projected_bundle["metadata"] = dict(native["metadata"])
projected_bundle["metadata"].update({
    "artifact_type": "projected_active_positive_with_native_cfg_negative_conditioning",
    "source_model": SOURCE_MODEL_ID,
    "source_representation": "raw_language_model_input_embedding_lookup",
    "alignment": "SEAM_character_overlap_weighted_occurrence_aware",
    "projector_format": projector_bundle["format"],
    "projector_training_pair_count": len(X_cpu),
    "projector_training_prompt_count": len(prompts),
    "projector_uses_same_baseline_prompt_set": True,
    "native_cfg_negative_preserved": True,
})

torch.save(projector_bundle, PROJECTOR_PATH)
torch.save(projected_bundle, PROJECTED_PATH)

saved = torch.load(PROJECTED_PATH, map_location="cpu", weights_only=True)
assert saved["format"] == projected_bundle["format"]
assert saved["active_prompt_embeds"].shape == target_active.shape
assert torch.equal(saved["prompt_attention_mask"], native["prompt_attention_mask"])
assert torch.equal(saved["prompt_input_ids"], native["prompt_input_ids"])
assert torch.equal(saved["prompt_row_offsets"], native["prompt_row_offsets"])
assert torch.equal(
    saved["active_negative_prompt_embeds"],
    native["active_negative_prompt_embeds"],
)
assert saved["prompts"] == native["prompts"]

print("Projector:", PROJECTOR_PATH)
print("Projected conditioning:", PROJECTED_PATH)
print("Projector size MiB:", round(PROJECTOR_PATH.stat().st_size / 1024**2, 1))
print("Projected size MiB:", round(PROJECTED_PATH.stat().st_size / 1024**2, 1))

display(FileLink(str(PROJECTOR_PATH)))
display(FileLink(str(PROJECTED_PATH)))

# %% [markdown]
# ## Final native-versus-projected token report

# %%
def scale_summary(name, tensor):
    norms = tensor.float().norm(dim=1)
    print(
        f"{name:28s}",
        f"min={norms.min().item():.3f}",
        f"median={norms.median().item():.3f}",
        f"mean={norms.mean().item():.3f}",
        f"max={norms.max().item():.3f}",
    )


scale_summary("Native active tokens", target_active)
scale_summary("Projected active tokens", saved["active_prompt_embeds"])

final_token_metrics = projection_metrics(
    saved["active_prompt_embeds"].float().to(fit_device),
    target_active.float().to(fit_device),
)

print("Final norm metrics:", norm_metrics(
    saved["active_prompt_embeds"],
    target_active,
))
print("Fallback count:", int(pairs["alignment_fallback_mask"].sum()))
print("Unique fitted target IDs:", len(fit_token_ids))
print("Projected prompt count:", len(saved["prompts"]))

