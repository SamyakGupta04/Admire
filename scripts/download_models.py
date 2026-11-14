#!/usr/bin/env python
"""Pre-download the model weights (about 5.5 GB) into .hf_cache/ inside the project.

    python scripts/download_models.py            # all three
    python scripts/download_models.py faces      # SD 1.5 inpainting only
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache"))
from huggingface_hub import snapshot_download  # noqa: E402

FP16 = ["model_index.json", "*/config.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
        "unet/*.fp16.safetensors", "vae/*.fp16.safetensors", "text_encoder/*.fp16.safetensors"]
MODELS = {
    "faces": [("stable-diffusion-v1-5/stable-diffusion-inpainting", FP16)],
    "monuments": [("sd2-community/stable-diffusion-2-inpainting", FP16),
                  ("thibaud/controlnet-sd21-canny-diffusers", None)],
}

if __name__ == "__main__":
    for group in sys.argv[1:] or list(MODELS):
        for repo, patterns in MODELS[group]:
            print("downloading", repo, flush=True)
            print("  ->", snapshot_download(repo, allow_patterns=patterns), flush=True)
    print("MODELS_DONE")
