import argparse
import json
import os
import sys
import threading
import time
import webbrowser

import psutil
from flask import Flask, Response, jsonify, render_template, request
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from conhecimento_rag import (
    adicionar_conhecimento_local,
    carregar_conhecimento_local,
    pesquisar_conhecimento,
)

# =====================================================================
# CONFIGURAÇÕES DA MILA AI (OTIMIZADA PARA RYZEN 5 5600G / 16GB RAM)
# =====================================================================
NOME_DA_IA = "Mila AI"
CRIADOR = "Artur"

# Catálogo de Cérebros GGUF 4-Bit (Q4_K_M) - Muito mais inteligentes e 3x mais leves que Float32
MODELOS_DISPONIVEIS = {
    "3b": {
        "id": "3b",
        "nome": "Qwen 2.5 - 3B Instruct (4-Bit GGUF) [Recomendado]",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "ram_aprox": "~2.1 GB",
        "descricao": "Equilíbrio ideal: raciocínio elaborado em português, rápido na CPU e baixo uso de RAM.",
    },
    "7b": {
        "id": "7b",
        "nome": "Qwen 2.5 - 7B Instruct (4-Bit GGUF) [Ultra Inteligente]",
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "ram_aprox": "~4.6 GB",
        "descricao": "Máxima inteligência e profundidade analítica para perguntas complexas.",
    },
    "1.5b": {
        "id": "1.5b",
        "nome": "Qwen 2.5 - 1.5B Instruct (4-Bit GGUF) [Modo Turbo]",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "ram_aprox": "~1.1 GB",
        "descricao": "Ultra leve e instantâneo, consumindo apenas ~1 GB de memória.",
    },
}

INSTRUCAO_SISTEMA_BASE = f"""O seu nome é {NOME_DA_IA}. Você é uma assistente de inteligência artificial altamente inteligente, lógica, culta, analítica e prestativa, criada pelo {CRIADOR}.

DIRETRIZES DE RESPOSTA:
1. Idioma: Responda SEMPRE em português fluente, natural e correto.
2. Profundidade e Estrutura: Quando o utilizador fizer perguntas elaboradas, explique o tema com profundidade real. Organize a explicação de forma clara (usando introdução, tópicos em negrito, exemplos práticos e conclusão quando apropriado).
3. Fidelidade aos Fatos: Quando houver uma seção [BASE DE CONHECIMENTO VERIFICADA] na mensagem, utilize esses dados reais (Wikipedia/Web/Memória) para fundamentar sua resposta com precisão, datas, conceitos técnicos e fatos reais, sem inventar informações.
4. Continuidade: Se o utilizador pedir para desenvolver ou aprofundar a resposta anterior, expanda o último tema abordado trazendo novos detalhes técnicos, históricos ou práticos."""


class MotorMilaAI:
    def __init__(self, modelo_padrao="3b", n_threads=4):
        self.modelo_atual_id = modelo_padrao
        self.n_threads = n_threads  # 4 threads deixa 2 núcleos livres no Ryzen 5 5600G
        self.usar_rag = True
        self.max_new_tokens = 900  # Permite respostas longas e completas (antes era 200)
        self.temperature = 0.25
        self.llm = None
        self.carregando = False
        self.status_carregamento = "A iniciar..."
        self.lock = threading.Lock()
        self.historico = [{"role": "system", "content": INSTRUCAO_SISTEMA_BASE}]

    def _localizar_gguf(self, info):
        """Usa o arquivo já baixado (sem internet); só baixa se ainda não existir."""
        try:
            return hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                local_files_only=True,
            )
        except Exception:
            self.status_carregamento = f"A transferir {info['nome']} ({info['ram_aprox']}) — só na 1ª vez..."
            print(f"[*] {self.status_carregamento}", flush=True)
            return hf_hub_download(repo_id=info["repo_id"], filename=info["filename"])

    def carregar_modelo(self, modelo_id=None, n_threads=None):
        # Evita duas recargas ao mesmo tempo (ex.: clicar "Aplicar" várias vezes)
        if self.carregando:
            return False

        self.carregando = True
        try:
            if modelo_id and modelo_id in MODELOS_DISPONIVEIS:
                self.modelo_atual_id = modelo_id
            if n_threads:
                self.n_threads = max(1, min(12, int(n_threads)))

            info = MODELOS_DISPONIVEIS[self.modelo_atual_id]
            self.status_carregamento = f"A verificar {info['nome']}..."
            print(f"\n[*] {self.status_carregamento}", flush=True)

            caminho_gguf = self._localizar_gguf(info)
            self.status_carregamento = f"A carregar motor GGUF 4-bit na CPU ({self.n_threads} threads)..."
            print(f"[*] {self.status_carregamento}", flush=True)

            with self.lock:
                # Libera modelo anterior da RAM antes de carregar o novo
                if self.llm is not None:
                    del self.llm
                    self.llm = None

                self.llm = Llama(
                    model_path=caminho_gguf,
                    n_ctx=4096,               # Contexto amplo para conversas + artigos da Wikipedia
                    n_threads=self.n_threads, # Controle de núcleos para não travar o PC
                    n_threads_batch=self.n_threads,
                    n_batch=512,              # Leitura rápida do prompt
                    use_mmap=True,            # Mapeamento inteligente de memória
                    verbose=False,
                )

            self.status_carregamento = "Pronta"
            print(f"[OK] {NOME_DA_IA} pronta com {info['nome']} usando {self.n_threads} threads!\n", flush=True)
            return True
        except Exception as e:
            self.status_carregamento = f"Erro ao carregar modelo: {e}"
            print(f"[ERRO] {self.status_carregamento}", flush=True)
            return False
        finally:
            self.carregando = False

    def limpar_historico(self):
        self.historico = [{"role": "system", "content": INSTRUCAO_SISTEMA_BASE}]

    def preparar_mensagens(self, pergunta: str):
        """Realiza busca RAG (se ativa) e monta o histórico de mensagens otimizado."""
        fontes = []
        termo_pesquisado = ""
        contexto_rag = ""

        perguntas_usuario = [
            m["content"] for m in self.historico if m["role"] == "user"
        ]

        if self.usar_rag:
            resultado_rag = pesquisar_conhecimento(pergunta, perguntas_usuario)
            termo_pesquisado = resultado_rag["termo"]
            contexto_rag = resultado_rag["contexto_formatado"]
            fontes = resultado_rag["fontes"]

        self.historico.append({"role": "user", "content": pergunta})
        # Mantém instrução do sistema + últimas 8 mensagens para não estourar a janela de 4096 tokens
        historico_curto = [self.historico[0]] + [
            dict(m) for m in self.historico[-8:]
        ]

        if contexto_rag:
            # Injeta o conhecimento verificado apenas no turno atual para o modelo consultar
            historico_curto[-1] = {
                "role": "user",
                "content": (
                    f"[BASE DE CONHECIMENTO VERIFICADA (Wikipedia / Web / Memória)]:\n"
                    f"{contexto_rag}\n\n"
                    f"---\n"
                    f"PERGUNTA DO UTILIZADOR: {pergunta}\n"
                    f"(Responda de forma elaborada, bem estruturada e precisa em português, "
                    f"utilizando os fatos relevantes acima e o seu próprio conhecimento lógico.)"
                ),
            }

        return historico_curto, fontes, termo_pesquisado

    def gerar_resposta_stream(self, pergunta: str):
        """Gera a resposta token a token (usado tanto na Web quanto no Terminal)."""
        if self.llm is None or self.carregando:
            yield {
                "tipo": "erro",
                "texto": f"O modelo ainda está a carregar ({self.status_carregamento}). Aguarde alguns segundos.",
            }
            return

        mensagens, fontes, termo = self.preparar_mensagens(pergunta)
        yield {
            "tipo": "meta",
            "fontes": fontes,
            "termo": termo,
        }

        resposta_completa = []
        inicio = time.time()
        tokens_gerados = 0

        with self.lock:
            fluxo = self.llm.create_chat_completion(
                messages=mensagens,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                repeat_penalty=1.12,
                stream=True,
            )
            for chunk in fluxo:
                delta = chunk["choices"][0].get("delta", {})
                pedaco = delta.get("content", "")
                if pedaco:
                    resposta_completa.append(pedaco)
                    tokens_gerados += 1
                    yield {"tipo": "token", "texto": pedaco}

        duracao = max(0.1, time.time() - inicio)
        velocidade = round(tokens_gerados / duracao, 1)
        texto_final = "".join(resposta_completa).strip()
        self.historico.append({"role": "assistant", "content": texto_final})

        yield {
            "tipo": "fim",
            "tokens": tokens_gerados,
            "tempo_s": round(duracao, 1),
            "tokens_por_s": velocidade,
        }


# =====================================================================
# INTERFACE VISUAL WEB (FLASK + STREAMING + MONITOR DE HARDWARE)
# =====================================================================
# O layout visual fica em templates/mila.html e o ícone em static/mila_robo.svg

app = Flask(__name__)
motor = MotorMilaAI(modelo_padrao="3b", n_threads=4)
processo_atual = psutil.Process(os.getpid())


@app.route("/")
def index():
    return render_template(
        "mila.html",
        nome_ia=NOME_DA_IA,
        criador=CRIADOR,
        modelos=MODELOS_DISPONIVEIS,
        modelo_atual=motor.modelo_atual_id,
        n_threads=motor.n_threads,
        status_inicial=motor.status_carregamento,
    )


@app.route("/api/status")
def api_status():
    ram_gb = round(processo_atual.memory_info().rss / (1024**3), 2)
    cpu_pct = psutil.cpu_percent(interval=None)
    return jsonify(
        {
            "status": motor.status_carregamento,
            "carregando": motor.carregando,
            "modelo_id": motor.modelo_atual_id,
            "modelo_nome": MODELOS_DISPONIVEIS[motor.modelo_atual_id]["nome"],
            "n_threads": motor.n_threads,
            "ram_processo_gb": ram_gb,
            "cpu_percent": cpu_pct,
        }
    )


@app.route("/api/opcoes", methods=["POST"])
def api_opcoes():
    dados = request.get_json(force=True)
    motor.usar_rag = bool(dados.get("usar_rag", True))
    motor.max_new_tokens = int(dados.get("max_tokens", 950))
    return jsonify({"ok": True})


@app.route("/api/trocar_modelo", methods=["POST"])
def api_trocar_modelo():
    dados = request.get_json(force=True)
    modelo_id = dados.get("modelo_id", "3b")
    n_threads = dados.get("n_threads", 4)
    threading.Thread(
        target=motor.carregar_modelo, args=(modelo_id, n_threads), daemon=True
    ).start()
    return jsonify({"ok": True})


@app.route("/api/memoria", methods=["POST"])
def api_memoria():
    dados = request.get_json(force=True)
    adicionar_conhecimento_local(dados.get("topico", ""), dados.get("conteudo", ""))
    return jsonify({"ok": True})


@app.route("/api/limpar", methods=["POST"])
def api_limpar():
    motor.limpar_historico()
    return jsonify({"ok": True})


@app.route("/api/chat_stream", methods=["POST"])
def api_chat_stream():
    dados = request.get_json(force=True)
    pergunta = dados.get("pergunta", "").strip()

    def gerar():
        for evento in motor.gerar_resposta_stream(pergunta):
            yield json.dumps(evento, ensure_ascii=False) + "\n"

    return Response(gerar(), mimetype="application/x-ndjson")


# =====================================================================
# MODO TERMINAL INTERATIVO (OPCIONAL: python teste_ia.py --terminal)
# =====================================================================
def executar_modo_terminal():
    print("\n" + "=" * 65)
    print(f"  {NOME_DA_IA} 2.0 (Motor GGUF 4-Bit + Wikipedia/Web RAG)")
    print("  Comandos: 'sair' para fechar | '/rag' liga/desliga busca web")
    print("=" * 65 + "\n")

    motor.carregar_modelo()

    while True:
        try:
            pergunta = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not pergunta:
            continue
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
        if pergunta.lower() == "/rag":
            motor.usar_rag = not motor.usar_rag
            print(f"[*] Busca Wikipedia/Web RAG: {'ATIVADA' if motor.usar_rag else 'DESATIVADA'}")
            continue

        for evento in motor.gerar_resposta_stream(pergunta):
            if evento["tipo"] == "meta":
                if evento["fontes"]:
                    nomes_fontes = ", ".join(f["fonte"] for f in evento["fontes"][:3])
                    print(f"  [Fontes consultadas: {nomes_fontes}]")
                print(f"\n{NOME_DA_IA}: ", end="", flush=True)
            elif evento["tipo"] == "token":
                print(evento["texto"], end="", flush=True)
            elif evento["tipo"] == "fim":
                print(
                    f"\n\n  [{evento['tokens']} tokens | {evento['tempo_s']}s | {evento['tokens_por_s']} t/s]"
                )
                print("-" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{NOME_DA_IA} - Assistente Inteligente GGUF + RAG")
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Executa apenas no modo terminal (sem abrir a interface visual Web).",
    )
    parser.add_argument(
        "--modelo",
        default="3b",
        choices=["1.5b", "3b", "7b"],
        help="Escolhe o modelo GGUF inicial (1.5b, 3b ou 7b). Padrão: 3b",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Número de threads da CPU (Padrão: 4 para Ryzen 5 5600G).",
    )
    parser.add_argument(
        "--porta",
        type=int,
        default=5000,
        help="Porta da interface Web (Padrão: 5000).",
    )
    args = parser.parse_args()

    motor.modelo_atual_id = args.modelo
    motor.n_threads = args.threads

    if args.terminal:
        executar_modo_terminal()
    else:
        # Carrega o modelo em background enquanto abre a interface visual
        threading.Thread(target=motor.carregar_modelo, daemon=True).start()
        url = f"http://127.0.0.1:{args.porta}"
        print("\n" + "=" * 65)
        print(f"  >> {NOME_DA_IA} iniciada com Interface Visual Web!")
        print(f"  >> Acesse no navegador: {url}")
        print("  >> Dica: Para usar apenas no terminal, rode: python teste_ia.py --terminal")
        print("=" * 65 + "\n")
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
        app.run(host="127.0.0.1", port=args.porta, debug=False, threaded=True)
