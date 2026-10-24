# 💚 Mila AI 2.0 — Assistente Inteligente Local (GGUF 4-Bit + RAG)

A **Mila AI** é uma assistente de inteligência artificial local criada por **Artur Lima**, projetada para responder perguntas elaboradas com alto nível de conhecimento sem sobrecarregar o hardware do computador.

---

## ✨ Principais Recursos

- **⚡ Motor Quantizado em 4-Bit GGUF (`llama-cpp-python`)**:
  - Executa modelos **Qwen 2.5 Instruct (1.5B, 3B ou 7B)** quantizados em 4 bits (`Q4_K_M`), consumindo até **3x menos memória RAM** que modelos em Float32 e rodando de forma leve na CPU (otimizado para processadores como o **AMD Ryzen 5 5600G**).
- **🌐 Busca de Conhecimento em Tempo Real (RAG)**:
  - Consulta automaticamente a **Wikipedia (PT)** e a **Web (DuckDuckGo)** em paralelo antes de responder perguntas complexas, citando as fontes consultadas.
- **🌱 Memória Local Permanente**:
  - Permite ensinar novos fatos e conhecimentos personalizados que ficam salvos em `conhecimento_local.json`.
- **🎨 Interface Visual Verde Pastel + Ícones SVG Exclusivos**:
  - Interface Web moderna com streaming em tempo real, renderização de Markdown, monitor de CPU/RAM ao vivo, seletor de cérebro (1.5B / 3B / 7B) e mascote robô exclusivo em SVG.
  - Também inclui **Modo Terminal** (`python teste_ia.py --terminal`).

---

## 🚀 Como Instalar e Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Iniciar a Mila AI (Interface Visual Web)
```bash
python teste_ia.py
```
Ou dê dois cliques em **`iniciar_mila.bat`** no Windows. O navegador abrirá automaticamente em `http://127.0.0.1:5000`.

### 3. Modo Terminal (Opcional)
```bash
python teste_ia.py --terminal
```

---

## 📁 Estrutura do Projeto

- `teste_ia.py` — Servidor principal, motor GGUF (`MotorMilaAI`) e modo terminal.
- `conhecimento_rag.py` — Motor de pesquisa em tempo real (Wikipedia PT + Web + Memória Local).
- `templates/mila.html` — Interface visual em tons de verde pastel com ícones SVG desenhados sob medida.
- `static/mila_robo.svg` — Ícone e mascote oficial da Mila AI.
- `conhecimento_local.json` — Base de conhecimento local personalizável.
- `iniciar_mila.bat` — Atalho rápido para iniciar no Windows.

---

Feito com 💚 por **Artur Lima**.
