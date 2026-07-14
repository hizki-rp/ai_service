"""
Rezumae AI Service — SmolLM2-135M-Instruct (HuggingFace transformers, CPU)
No GGUF needed. Model downloads automatically on first start (~270 MB).
"""

import json
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline as hf_pipeline

MODEL_NAME      = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://rezumae.pro,http://localhost:5173").split(",")

pipe = None   # loaded once at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print(f"[startup] Loading {MODEL_NAME} on CPU …", flush=True)
    t0 = time.time()
    pipe = hf_pipeline(
        "text-generation",
        model=MODEL_NAME,
        device="cpu",
        dtype="bfloat16",   # halves weight memory: ~270MB instead of ~540MB
    )
    print(f"[startup] Ready in {time.time() - t0:.1f}s", flush=True)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class ParseCvRequest(BaseModel):
    text: str

class AiCompleteRequest(BaseModel):
    messages: list
    max_tokens: int = 512
    temperature: float = 0.4


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    # Truncated output — try to close open braces/arrays
    m2 = re.search(r"\{", text)
    if m2:
        fragment = text[m2.start():]
        opens = fragment.count("{") - fragment.count("}")
        arr_opens = fragment.count("[") - fragment.count("]")
        fragment += "]" * max(arr_opens, 0) + "}" * max(opens, 0)
        try:
            return json.loads(fragment)
        except Exception:
            pass
    raise ValueError("No JSON found in model output")


def run_chat(messages: list, max_new_tokens: int, temperature: float) -> str:
    """Run the pipeline and return the assistant's reply text."""
    out = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=max(temperature, 0.01),
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/parse-cv")
def parse_cv(req: ParseCvRequest):
    if pipe is None:
        raise HTTPException(503, "Model not loaded")

    text = req.text.strip()[:3000]   # trim so we stay inside context
    if not text:
        raise HTTPException(400, "Empty CV text")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise resume parser. Output ONLY valid JSON, no markdown.\n"
                "Keys: name, email, phone, location, title, summary, "
                "experience (array of {role,company,startDate,endDate,description}), "
                "education (array of {degree,school,endDate}), "
                "skills (array of strings)."
            ),
        },
        {"role": "user", "content": f"Parse this resume:\n\n{text}"},
    ]

    t0 = time.time()
    raw = run_chat(messages, max_new_tokens=600, temperature=0.05)
    elapsed = round(time.time() - t0, 2)

    try:
        data = extract_json(raw)
        return {"ok": True, "data": data, "elapsed": elapsed}
    except Exception as e:
        return {"ok": False, "raw": raw, "error": str(e), "elapsed": elapsed}


@app.post("/ai-complete")
def ai_complete(req: AiCompleteRequest):
    """OpenAI-compatible completion endpoint — used as last-resort fallback."""
    if pipe is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.time()
    content = run_chat(req.messages, max_new_tokens=min(req.max_tokens, 512), temperature=req.temperature)
    elapsed = round(time.time() - t0, 2)

    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "elapsed": elapsed,
    }
