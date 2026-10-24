@echo off
chcp 65001 >nul
title Mila AI 2.0 - Motor GGUF 4-Bit + RAG
cd /d "%~dp0"
echo =================================================================
echo   Iniciando Mila AI 2.0 (Motor GGUF 4-Bit + Wikipedia/Web RAG)
echo =================================================================
if exist "BitNet\venv\Scripts\python.exe" (
    "BitNet\venv\Scripts\python.exe" teste_ia.py %*
) else (
    python teste_ia.py %*
)
pause
