"""
Anima LoKr Loader for ComfyUI.

Loads Anima-format LoKr checkpoints that use the current anima_lora
`lora_unet_*` key layout, with lightweight block presets and optional
per-block strength overrides.
"""

import os
import re

import comfy.sd
import comfy.utils
import folder_paths
from safetensors import safe_open


ANIMA_MAIN_BLOCKS = tuple(f"block_{i}" for i in range(28))
ANIMA_LLM_ADAPTER_BLOCKS = tuple(f"llm_adapter_{i}" for i in range(6))
ANIMA_SPECIAL_BLOCKS = (
    "llm_adapter_io",
    "final_layer",
    "t_embedder",
    "x_embedder",
    "other_weights",
)
ANIMA_BLOCKS = (
    *ANIMA_MAIN_BLOCKS,
    *ANIMA_LLM_ADAPTER_BLOCKS,
    *ANIMA_SPECIAL_BLOCKS,
)
ANIMA_BLOCK_SET = frozenset(ANIMA_BLOCKS)

ANIMA_PRESETS = {
    "Default": ANIMA_BLOCKS,
    "All Off": (),
    "Half Strength": ANIMA_BLOCKS,
    "Main Blocks Only": (
        *ANIMA_MAIN_BLOCKS,
        "final_layer",
        "t_embedder",
        "x_embedder",
        "other_weights",
    ),
    "LLM Adapter Only": (
        *ANIMA_LLM_ADAPTER_BLOCKS,
        "llm_adapter_io",
        "other_weights",
    ),
    "Late Main (20-27)": (
        *(f"block_{i}" for i in range(20, 28)),
        "final_layer",
        "t_embedder",
        "x_embedder",
        "other_weights",
    ),
    "Mid-Late Main (14-27)": (
        *(f"block_{i}" for i in range(14, 28)),
        "final_layer",
        "t_embedder",
        "x_embedder",
        "other_weights",
    ),
    "Evens Only": (
        *(f"block_{i}" for i in range(0, 28, 2)),
        *(f"llm_adapter_{i}" for i in range(0, 6, 2)),
    ),
    "Odds Only": (
        *(f"block_{i}" for i in range(1, 28, 2)),
        *(f"llm_adapter_{i}" for i in range(1, 6, 2)),
    ),
    "Custom": ANIMA_BLOCKS,
}
PRESET_NAMES = list(ANIMA_PRESETS.keys())

_LLM_BLOCK_RE = re.compile(r"(?:^|_)llm_adapter_blocks_(\d+)_")
_MAIN_BLOCK_RE = re.compile(r"(?:^|_)blocks_(\d+)_")


def _read_metadata(lora_path: str) -> dict:
    if not lora_path.endswith(".safetensors"):
        return {}
    try:
        with safe_open(lora_path, framework="pt") as handle:
            return handle.metadata() or {}
    except Exception as exc:
        print(f"[AnimaLoKrLoader] Could not read metadata: {exc}")
        return {}


def _normalize_strength(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    parsed = max(-2.0, min(2.0, parsed))
    return round(parsed, 3)


def _parse_blocks_csv(raw_value: str) -> list[str]:
    if not raw_value or not raw_value.strip():
        return []
    raw_items = [item.strip() for item in raw_value.split(",")]
    selected = {item for item in raw_items if item in ANIMA_BLOCK_SET}
    return [block for block in ANIMA_BLOCKS if block in selected]


def _parse_block_strengths(raw_value: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    if not raw_value or not raw_value.strip():
        return parsed
    for item in raw_value.split(","):
        if "=" not in item:
            continue
        key, raw_strength = item.split("=", 1)
        key = key.strip()
        if key not in ANIMA_BLOCK_SET:
            continue
        parsed[key] = _normalize_strength(raw_strength, default=0.0)
    return parsed


def _classify_anima_key(key: str) -> str:
    value = str(key or "").strip().lower()

    match = _LLM_BLOCK_RE.search(value)
    if match:
        return f"llm_adapter_{match.group(1)}"

    match = _MAIN_BLOCK_RE.search(value)
    if match:
        return f"block_{match.group(1)}"

    if (
        "llm_adapter_embed_" in value
        or "llm_adapter_norm_" in value
        or "llm_adapter_out_proj_" in value
    ):
        return "llm_adapter_io"
    if "final_layer_" in value:
        return "final_layer"
    if "t_embedder_" in value or "t_embedding_norm_" in value:
        return "t_embedder"
    if "x_embedder_" in value:
        return "x_embedder"
    return "other_weights"


def _is_lokr_checkpoint(state_dict: dict, metadata: dict) -> bool:
    network_spec = str(metadata.get("ss_network_spec") or "").strip().lower()
    if network_spec == "lokr":
        return True
    for key in state_dict.keys():
        key_lower = key.lower()
        if ".lokr_w1" in key_lower or ".lokr_w2" in key_lower:
            return True
    return False


def _build_block_strengths(
    preset: str,
    enabled_blocks: str,
    block_strengths: str,
    global_strength: float,
) -> dict[str, float]:
    if enabled_blocks and enabled_blocks.strip():
        selected = set(_parse_blocks_csv(enabled_blocks))
    else:
        selected = set(ANIMA_PRESETS.get(preset, ANIMA_PRESETS["Default"]))

    strengths = {
        block: (1.0 if block in selected else 0.0)
        for block in ANIMA_BLOCKS
    }

    if preset == "Half Strength":
        for block, value in list(strengths.items()):
            if value != 0.0:
                strengths[block] = 0.5

    for block, value in _parse_block_strengths(block_strengths).items():
        strengths[block] = value

    return {
        block: round(value * global_strength, 3)
        for block, value in strengths.items()
    }


def _filter_anima_lokr(state_dict: dict, block_strengths: dict[str, float]) -> tuple[dict, set[str]]:
    filtered = {}
    matched_blocks: set[str] = set()

    for key, value in state_dict.items():
        block_id = _classify_anima_key(key)
        scale = block_strengths.get(block_id, 0.0)
        if scale == 0.0:
            continue

        key_lower = key.lower()
        should_scale = (
            (".lokr_w1" in key_lower and ".lokr_w1_b" not in key_lower)
            and ".lokr_w2" not in key_lower
            and ".alpha" not in key_lower
        )
        filtered[key] = value * scale if should_scale and scale != 1.0 else value
        matched_blocks.add(block_id)

    return filtered, matched_blocks


class AnimaLoKrLoader:
    """Load Anima-format LoKr checkpoints using anima_lora key routing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Anima LoKr safetensors file from the loras directory."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Global multiplier for the selected LoKr blocks."
                }),
                "preset": (PRESET_NAMES, {
                    "default": "Default",
                    "tooltip": "Quick block selection preset."
                }),
                "enabled_blocks": ("STRING", {
                    "default": "",
                    "tooltip": "Optional comma-separated block list. When filled, overrides the preset selection."
                }),
                "block_strengths": ("STRING", {
                    "default": "",
                    "tooltip": "Optional per-block overrides like block_22=0.8,final_layer=1.2."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_lokr"
    CATEGORY = "loaders/lora"
    DESCRIPTION = (
        "Loads Anima-format LoKr weights that use the current anima_lora "
        "lora_unet_* key layout. Uses Anima block presets and only scales "
        "lokr_w1 during filtering, so LoKr strength is not accidentally squared."
    )

    def load_lokr(self, model, lora_name, strength, preset, enabled_blocks="", block_strengths=""):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            report = f"LoKr not found: {lora_name}"
            print(f"[AnimaLoKrLoader] {report}")
            return (model, report)

        metadata = _read_metadata(lora_path)
        state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)

        if not _is_lokr_checkpoint(state_dict, metadata):
            report = "Selected file is not recognized as a LoKr checkpoint."
            print(f"[AnimaLoKrLoader] {report}")
            return (model, report)

        if strength == 0:
            report = "Strength is 0, skipped loading."
            print(f"[AnimaLoKrLoader] {report}")
            return (model, report)

        effective_strengths = _build_block_strengths(
            preset,
            enabled_blocks,
            block_strengths,
            float(strength),
        )
        filtered_lora, matched_blocks = _filter_anima_lokr(state_dict, effective_strengths)

        if not filtered_lora:
            report = "No Anima LoKr tensors matched the requested blocks."
            print(f"[AnimaLoKrLoader] {report}")
            return (model, report)

        model_out, _ = comfy.sd.load_lora_for_models(
            model,
            None,
            filtered_lora,
            1.0,
            0.0,
        )

        active_blocks = [block for block in ANIMA_BLOCKS if effective_strengths.get(block, 0.0) != 0.0]
        report = (
            f"Loaded {lora_name}: {len(filtered_lora)}/{len(state_dict)} tensors, "
            f"{len(matched_blocks)} matched blocks, {len(active_blocks)} requested blocks."
        )
        print(f"[AnimaLoKrLoader] {report}")
        return (model_out, report)


NODE_CLASS_MAPPINGS = {
    "AnimaLoKrLoader": AnimaLoKrLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoKrLoader": "Anima LoKr Loader",
}
