"""
Downloads SmolLM2-135M-Instruct Q4_K_M GGUF into the working directory.
Run during Render build: python download_model.py
"""
import os, time
from huggingface_hub import hf_hub_download

REPO_ID  = os.environ.get("GGUF_REPO", "bartowski/SmolLM2-135M-Instruct-GGUF")
FILENAME = os.environ.get("GGUF_FILE", "SmolLM2-135M-Instruct-Q4_K_M.gguf")

print(f"Downloading {FILENAME} from {REPO_ID} …")
t0 = time.time()
path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=".")
print(f"Done in {time.time()-t0:.1f}s — saved to {path}")
