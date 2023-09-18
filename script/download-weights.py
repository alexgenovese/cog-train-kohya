import os
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

BASE_MODEL_CACHE = "./base-model-cache"

print("Checking model cache...")

# BASE MODEL CHECKER
if not os.path.exists(BASE_MODEL_CACHE):
    # Trying to use StableDiffusionXLPipeline instead of DiffusionPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.save_pretrained(BASE_MODEL_CACHE, safe_serialization=True)


print("Everything is here")