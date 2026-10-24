import json
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from html import unescape
import requests

ARQUIVO_CONHECIMENTO_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "conhecimento_local.json"
)

HEADERS = {
    "User-Agent": "MilaAI/2.0 (Assistente Inteligente criada por Artur; Python/requests)"
}

# Palavras comuns para limpar a pergunta e extrair o tópico exato de pesquisa
PREFIXOS_PERGUNTA = [
    r"^(?:mila\s*,?\s*)?",
    r"^(?:por favor\s*,?\s*)?",
    r"^(?:você pode(?:ria)?\s+me\s+(?:explicar|dizer|falar|contar|ensinar)\s+)?",
    r"^(?:me\s+(?:explique|diga|fale|conte|ensine|ajude\s+a\s+entender)\s+)?",
    r"^(?:explique(?:-me)?\s+|fale(?:-me)?\s+sobre\s+|conte(?:-me)?\s+sobre\s+|discorra\s+sobre\s+)?",
    r"^(?:de\s+forma\s+(?:elaborada|detalhada|profunda|completa|clara|simples)\s*,?\s*)?",
    r"^(?:o\s+que\s+(?:é|são|foi|foram|significa|significam)\s+)?",
    r"^(?:quem\s+(?:é|foi|são|foram)\s+)?",
    r"^(?:como\s+(?:funciona|funcionam|surgiu|aconteceu|fazer|se\s+faz)\s+)?",
    r"^(?:quais\s+(?:são|foram)\s+(?:os|as)\s+(?:principais\s+)?)?",
    r"^(?:qual\s+(?:é|foi)\s+(?:a|o)\s+(?:história|origem|importância|impacto|diferença)\s+(?:de|da|do|dos|das)\s+)?",
    r"^(?:sobre\s+)?",
]

SAUDACOES_CASUAIS = {
    "oi", "ola", "olá", "bom dia", "boa tarde", "boa noite", "tudo bem",
    "como vai", "quem é você", "quem e voce", "qual o seu nome", "qual seu nome",
    "quem criou você", "quem te criou", "obrigado", "obrigada", "valeu", "tchau"
}

PEDIDOS_CONTINUACAO = [
    "desenvolva", "explique melhor", "fale mais", "aprofunde", "mais detalhes",
    "continue", "detalhe mais", "explique com mais detalhes", "pode desenvolver"
]


def carregar_conhecimento_local():
    """Carrega fatos personalizados salvos pelo utilizador."""
    if not os.path.exists(ARQUIVO_CONHECIMENTO_LOCAL):
        dados_iniciais = [
            {
                "topico": "Mila AI / Identidade",
                "conteudo": (
                    "A Mila AI é uma assistente de inteligência artificial criada pelo Artur. "
                    "Ela roda localmente de forma otimizada usando quantização GGUF de 4 bits "
                    "(baseada na arquitetura llama.cpp / BitNet) com busca de conhecimento em tempo real."
                ),
            }
        ]
        try:
            with open(ARQUIVO_CONHECIMENTO_LOCAL, "w", encoding="utf-8") as f:
                json.dump(dados_iniciais, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return dados_iniciais

    try:
        with open(ARQUIVO_CONHECIMENTO_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def adicionar_conhecimento_local(topico: str, conteudo: str):
    """Salva um novo conhecimento permanente no arquivo local."""
    dados = carregar_conhecimento_local()
    dados.append({"topico": topico.strip(), "conteudo": conteudo.strip()})
    with open(ARQUIVO_CONHECIMENTO_LOCAL, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    return True


def eh_conversa_casual(pergunta: str) -> bool:
    """Verifica se é apenas um cumprimento rápido que não precisa de busca na web."""
    limpa = re.sub(r"[^\w\sáéíóúâêôãõç]", "", pergunta.lower()).strip()
    if len(limpa) < 4 or limpa in SAUDACOES_CASUAIS:
        return True
    return False


def extrair_termo_busca(pergunta: str, historico_usuario: list = None) -> str:
    """Extrai o tema principal da pergunta para pesquisar na Wikipedia e na Web."""
    texto = pergunta.strip()
    texto_lower = texto.lower()

    # Se o usuário pedir para desenvolver a resposta anterior, usa a pergunta anterior
    if historico_usuario and any(p in texto_lower for p in PEDIDOS_CONTINUACAO) and len(texto.split()) <= 8:
        for msg_antiga in reversed(historico_usuario[:-1]):
            if not any(p in msg_antiga.lower() for p in PEDIDOS_CONTINUACAO):
                texto = msg_antiga
                break

    termo = texto
    for padrao in PREFIXOS_PERGUNTA:
        termo = re.sub(padrao, "", termo, flags=re.IGNORECASE).strip()

    # Remove pontuações finais e termos genéricos no fim
    termo = re.sub(r"[?!.;,]+$", "", termo).strip()
    termo = re.sub(
        r"\s+(?:de forma elaborada|com detalhes|detalhadamente|por favor|passo a passo)$",
        "",
        termo,
        flags=re.IGNORECASE,
    ).strip()

    # Remove artigos soltos no início
    termo = re.sub(r"^(?:o|a|os|as|um|uma)\s+", "", termo, flags=re.IGNORECASE).strip()

    return termo if len(termo) >= 2 else texto


def buscar_wikipedia(termo: str, lang: str = "pt", limite_artigos: int = 2, max_chars: int = 1800) -> list:
    """Busca artigos e extrai conteúdo detalhado da Wikipedia."""
    url_api = f"https://{lang}.wikipedia.org/w/api.php"
    resultados = []
    try:
        resp_busca = requests.get(
            url_api,
            params={
                "action": "query",
                "list": "search",
                "srsearch": termo,
                "format": "json",
                "srlimit": limite_artigos,
            },
            headers=HEADERS,
            timeout=4.5,
        )
        if resp_busca.status_code != 200:
            return []

        itens = resp_busca.json().get("query", {}).get("search", [])
        if not itens:
            return []

        titulos = [item["title"] for item in itens]
        for idx, titulo_busca in enumerate(titulos):
            params_conteudo = {
                "action": "query",
                "prop": "extracts",
                "titles": titulo_busca,
                "explaintext": 1,
                "redirects": 1,
                "exsectionformat": "plain",
                "format": "json",
            }
            # No 2º artigo em diante, pega a introdução para ser mais rápido e focado
            if idx > 0:
                params_conteudo["exintro"] = 1

            resp_conteudo = requests.get(
                url_api,
                params=params_conteudo,
                headers=HEADERS,
                timeout=4.5,
            )
            if resp_conteudo.status_code != 200:
                continue

            paginas = resp_conteudo.json().get("query", {}).get("pages", {})
            for _, pagina in paginas.items():
                titulo = pagina.get("title", "")
                extrato = pagina.get("extract", "").strip()
                if not extrato:
                    continue
                extrato = re.sub(r"\n{3,}", "\n\n", extrato)
                limite_atual = max_chars if idx == 0 else max_chars // 2
                if len(extrato) > limite_atual:
                    extrato = extrato[:limite_atual].rsplit(" ", 1)[0] + "..."
                url_artigo = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(titulo.replace(' ', '_'))}"
                resultados.append(
                    {
                        "fonte": f"Wikipedia ({lang.upper()}): {titulo}",
                        "url": url_artigo,
                        "resumo": extrato,
                    }
                )
    except Exception:
        pass
    return resultados


def buscar_duckduckgo_web(termo: str, max_resultados: int = 3) -> list:
    """Busca trechos atualizados na web usando DuckDuckGo Lite."""
    resultados = []
    try:
        resp = requests.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": f"{termo}"},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            timeout=5.0,
        )
        if resp.status_code != 200:
            return []

        html = resp.text
        tags_a = re.findall(
            r'<a[^>]+class=["\']result-link["\'][^>]*>(.*?)</a>',
            html,
            flags=re.S | re.I,
        )
        hrefs_a = re.findall(
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]+class=["\']result-link["\']|<a[^>]+class=["\']result-link["\'][^>]+href=["\']([^"\']+)["\']',
            html,
            flags=re.S | re.I,
        )
        snippets = re.findall(
            r'<td[^>]+class=["\']result-snippet["\'][^>]*>(.*?)</td>',
            html,
            flags=re.S | re.I,
        )

        for i in range(min(len(snippets), max_resultados)):
            trecho = re.sub(r"<[^>]+>", "", snippets[i])
            trecho = unescape(re.sub(r"\s+", " ", trecho)).strip()
            if len(trecho) < 30:
                continue

            titulo = "Pesquisa Web"
            url = "https://duckduckgo.com"
            if i < len(tags_a):
                titulo = unescape(re.sub(r"<[^>]+>", "", tags_a[i])).strip()
            if i < len(hrefs_a):
                url = hrefs_a[i][0] or hrefs_a[i][1] or url

            resultados.append(
                {
                    "fonte": f"Web: {titulo[:65]}",
                    "url": url,
                    "resumo": trecho[:600],
                }
            )
    except Exception:
        pass
    return resultados


def pesquisar_conhecimento(pergunta: str, historico_usuario: list = None) -> dict:
    """
    Executa busca combinada (Memória Local + Wikipedia PT + Web) em paralelo.
    Retorna dicionário com 'termo', 'contexto_formatado' e lista de 'fontes'.
    """
    fontes_encontradas = []
    blocos_texto = []

    # 1. Verifica conhecimento local personalizado
    conhecimento_local = carregar_conhecimento_local()
    pergunta_lower = pergunta.lower()
    for item in conhecimento_local:
        palavras_topico = [
            p for p in re.split(r"\W+", item.get("topico", "").lower()) if len(p) > 2
        ]
        if any(p in pergunta_lower for p in palavras_topico):
            blocos_texto.append(f"[Memória Local - {item['topico']}]: {item['conteudo']}")
            fontes_encontradas.append(
                {"fonte": f"Memória Local: {item['topico']}", "url": "#memoria-local"}
            )

    if eh_conversa_casual(pergunta):
        return {
            "termo": "",
            "contexto_formatado": "\n\n".join(blocos_texto),
            "fontes": fontes_encontradas,
        }

    termo = extrair_termo_busca(pergunta, historico_usuario)

    # 2. Busca em paralelo na Wikipedia PT e na Web
    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_wiki_pt = executor.submit(buscar_wikipedia, termo, "pt", 2, 1600)
        fut_web = executor.submit(buscar_duckduckgo_web, termo, 3)

        wiki_pt = fut_wiki_pt.result()
        web_res = fut_web.result()

    # Se não achou na Wikipedia PT com o termo longo, tenta com as palavras-chave principais
    if not wiki_pt and len(termo.split()) > 3:
        palavras_fortes = [p for p in termo.split() if len(p) > 3][:3]
        if palavras_fortes:
            wiki_pt = buscar_wikipedia(" ".join(palavras_fortes), "pt", 2, 1500)

    for item in wiki_pt:
        blocos_texto.append(f"[{item['fonte']}]:\n{item['resumo']}")
        fontes_encontradas.append({"fonte": item["fonte"], "url": item["url"]})

    for item in web_res:
        blocos_texto.append(f"[{item['fonte']}]: {item['resumo']}")
        fontes_encontradas.append({"fonte": item["fonte"], "url": item["url"]})

    contexto_final = "\n\n".join(blocos_texto)
    # Limita tamanho total do contexto para manter a geração rápida na CPU
    if len(contexto_final) > 4200:
        contexto_final = contexto_final[:4200].rsplit(" ", 1)[0] + "..."

    return {
        "termo": termo,
        "contexto_formatado": contexto_final,
        "fontes": fontes_encontradas,
    }
