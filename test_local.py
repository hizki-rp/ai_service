"""
Quick sanity check before deploying to Render.
Run: python test_local.py
(requires the server to be running: uvicorn main:app --port 8000)
"""
import requests, json, time

BASE = "http://localhost:8001"

sample_cv = """
John Doe
Software Engineer
john@example.com | +1-555-123-4567 | New York, NY

SUMMARY
Backend engineer with 5 years building APIs in Django and Node.js.

EXPERIENCE
Backend Developer | Acme Corp | Jan 2021 – Present
- Built REST APIs handling 10k req/s using Django REST Framework
- Reduced DB query time by 40% with PostgreSQL indexes

Junior Dev | StartupXYZ | Jun 2019 – Dec 2020
- Built React dashboards
- Automated CI/CD with GitHub Actions

EDUCATION
BSc Computer Science | MIT | 2019

SKILLS
Python, Django, Node.js, React, PostgreSQL, Docker, Git
"""

print("── Health check ──────────────────────────────")
r = requests.get(f"{BASE}/health", timeout=5)
print(r.json())

print("\n── CV parse ──────────────────────────────────")
t0 = time.time()
r = requests.post(f"{BASE}/parse-cv", json={"text": sample_cv}, timeout=60)
elapsed = round(time.time() - t0, 2)
result = r.json()
print(f"HTTP {r.status_code}  ({elapsed}s)")
if result.get("ok"):
    print(json.dumps(result["data"], indent=2))
else:
    print("Parse failed:", result.get("error"))
    print("Raw output:", result.get("raw"))

print("\n── AI complete ───────────────────────────────")
r = requests.post(f"{BASE}/ai-complete", json={
    "messages": [
        {"role": "system", "content": "You are a resume assistant. Be concise."},
        {"role": "user",   "content": "Rewrite this bullet as a stronger action: 'helped with databases'"},
    ],
    "max_tokens": 100,
    "temperature": 0.4,
}, timeout=60)
print(f"HTTP {r.status_code}")
result = r.json()
print(result["choices"][0]["message"]["content"])
