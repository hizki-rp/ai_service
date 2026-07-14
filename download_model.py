"""
Pre-downloads SmolLM2-135M-Instruct into HuggingFace cache.
Run during Render build: python download_model.py
Locally: not needed — first uvicorn startup will download it automatically.
"""
import os, time

MODEL_NAME = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")

print(f"Pre-downloading {MODEL_NAME} …")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

AutoTokenizer.from_pretrained(MODEL_NAME)
AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print(f"Done in {time.time() - t0:.1f}s — model is cached.")
