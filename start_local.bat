@echo off
title Rezumae AI Service (SmolLM2-135M)
echo Starting Rezumae AI Service on http://localhost:8001
echo Model downloads automatically on first run (~270 MB from HuggingFace).
echo Subsequent starts are instant (cached).
echo Press Ctrl+C to stop.
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
pause
