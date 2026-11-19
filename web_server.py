"""
web_server.py — Interfaz HTTP ligera para el agente.

Uso:
    python web_server.py          # servidor por defecto en puerto 8080
    python web_server.py 9000     # puerto personalizado

Endpoints:
    POST /chat      → {"prompt": "tu mensaje"} → {"response": "respuesta del agente"}
    GET  /health    → {"status": "ok"}
    GET  /          → página HTML simple
"""

import json
import os
import sys
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Importaciones del proyecto
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import LLM, LLMError
from config import SISTEMA, logger
from tools import TOOL_FUNCTIONS
from schema import TOOLS_SCHEMA

# ---------------------------------------------------------------------------
# Estado global del agente (thread-safe)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_historial = []
_llm_instance = None


def _init_agent():
    """Inicializa el LLM y carga memoria previa."""
    global _llm_instance, _historial

    # Leer .env para obtener la API key del proveedor actual
    _vals = {}
    try:
        from dotenv import dotenv_values

        _vals = dotenv_values()
    except Exception:
        pass

    # Proveedor FreeLLMAPI (default)
    api_key = (
        _vals.get("FREELLM_API_KEY")
        or os.environ.get("FREELLM_API_KEY")
        or _vals.get("ORCAROUTER_API_KEY")
        or os.environ.get("ORCAROUTER_API_KEY")
        or _vals.get("GEMINI_API_KEY")
        or _vals.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Falta FREELLM_API_KEY (o ORCAROUTER_API_KEY, o GEMINI_API_KEY/GOOGLE_API_KEY)"
        )

    _llm_instance = LLM(
        proveedor="freellm",
        model=os.environ.get("FREELLM_MODEL", "auto:smart"),
        max_retries=1,
        base_delay=1.0,
        timeout_secs=float(os.environ.get("LLM_TIMEOUT_SECS", "60")),
    )

    # Cargar historial previo
    from memory import cargar_historial_previo, cargar_memoria

    memoria = cargar_memoria()
    sistema_completo = SISTEMA
    if memoria:
        recuerdos_str = "\n".join(f"- {k}: {v}" for k, v in memoria.items())
        sistema_completo += f"\nRecuerdos de sesiones anteriores:\n{recuerdos_str}\n"

    textos_previos = cargar_historial_previo()
    _historial = []
    if textos_previos:
        for t in textos_previos:
            if "text" in t and t["text"]:
                _historial.append(_llm_instance.crear_mensaje(t["role"], t["text"]))

    return sistema_completo


AGENTE_CONFIG = {
    "system_instruction": "",
    "tools": TOOLS_SCHEMA,
}


def _ejecutar_chat(prompt: str) -> str:
    """Procesa un mensaje del usuario y devuelve la respuesta."""
    global _historial

    if _llm_instance is None:
        _init_agent()

    with _lock:
        _historial.append(_llm_instance.crear_mensaje_usuario(prompt))

        MAX_ITER = 10
        iteraciones = 0
        respuesta_final = ""

        while iteraciones < MAX_ITER:
            iteraciones += 1
            try:
                # Usar no-streaming para evitar problemas con streaming de FreeLLMAPI
                response = _llm_instance.generar(
                    contents=_historial, config=AGENTE_CONFIG
                )
            except LLMError as e:
                logger.error(f"Error API: {e}")
                return f"[ERROR API] {e}"

            texto_completo = _llm_instance.extraer_texto(response)
            tiene_tool_calls = _llm_instance.tiene_tool_calls(response)

            if tiene_tool_calls:
                contenido = _llm_instance.extraer_contenido(response)
                if contenido:
                    _historial.append(contenido)

                tool_calls = _llm_instance.extraer_tool_calls(response)

                try:
                    for tc in tool_calls:
                        resultado = str(TOOL_FUNCTIONS[tc["name"]](**tc["args"]))
                        _historial.append(
                            _llm_instance.crear_mensaje_tool_response(
                                tc["name"], resultado
                            )
                        )
                except Exception as e:
                    logger.error(f"Error ejecutando herramienta: {e}")
                    texto_completo = f"[error ejecutando herramienta] {e}"

            if texto_completo:
                respuesta_final = texto_completo.strip()
                break

        if not respuesta_final:
            respuesta_final = "[aviso] No pude generar una respuesta."

        return respuesta_final


# ---------------------------------------------------------------------------
# Handler HTTP
# ---------------------------------------------------------------------------
class AgenteHandler(BaseHTTPRequestHandler):
    """Handler para el servidor HTTP del agente."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json_response({"status": "ok"})
        elif parsed.path == "/":
            self._serve_index()
        else:
            self._not_found()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/chat":
            self._handle_chat()
        else:
            self._not_found()

    def _handle_chat(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            prompt = data.get("prompt", "").strip()
            if not prompt:
                self._json_response({"error": "prompt vacio"}, 400)
                return

            # Ejecutar en hilo separado con timeout para que el servidor
            # nunca se cuelgue si Gemini no responde
            resultado = [None]
            excepcion = [None]

            def ejecutar():
                try:
                    resultado[0] = _ejecutar_chat(prompt)
                except Exception as e:
                    excepcion[0] = e

            hilo = threading.Thread(target=ejecutar, daemon=True)
            hilo.start()
            chat_timeout = int(os.environ.get("CHAT_TIMEOUT_SECS", "120"))
            hilo.join(timeout=chat_timeout)  # max configurable (default 120s)

            if excepcion[0]:
                raise excepcion[0]
            if resultado[0] is None:
                self._json_response(
                    {
                        "error": f"La solicitud excedió el tiempo límite ({chat_timeout}s)"
                    },
                    504,
                )
                return

            self._json_response({"response": resultado[0]})
        except Exception as e:
            logger.exception("Error en /chat")
            try:
                self._json_response({"error": str(e)}, 500)
            except Exception:
                pass  # conexión ya cerrada por el cliente

    def _serve_index(self):
        html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agente Fuse</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg-primary: #0a0a0f;
  --bg-secondary: #12121a;
  --bg-tertiary: #1a1a25;
  --bg-card: #16161f;
  --border: #2a2a3a;
  --border-light: #3a3a4a;
  --text-primary: #e8e8f0;
  --text-secondary: #a0a0b8;
  --text-muted: #6a6a80;
  --accent: #6366f1;
  --accent-hover: #818cf8;
  --accent-glow: rgba(99, 102, 241, 0.15);
  --user-bg: #1e1e2e;
  --agent-bg: #1a1a28;
  --error: #ef4444;
  --success: #22c55e;
  --radius: 12px;
  --radius-sm: 8px;
  --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
  --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background-image: 
    radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.05) 0%, transparent 50%),
    radial-gradient(ellipse at bottom right, rgba(139, 92, 246, 0.03) 0%, transparent 50%);
}

/* Header */
.header {
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
  background: var(--bg-secondary);
  backdrop-filter: blur(12px);
}

.header-logo {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, var(--accent), #8b5cf6);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 14px;
  color: white;
}

.header-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.header-subtitle {
  font-size: 12px;
  color: var(--text-muted);
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
}

/* Chat area */
#chat {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  scroll-behavior: smooth;
}

#chat::-webkit-scrollbar { width: 6px; }
#chat::-webkit-scrollbar-track { background: transparent; }
#chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Messages */
.msg {
  max-width: 75%;
  padding: 14px 18px;
  border-radius: var(--radius);
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-size: 14px;
  animation: fadeIn 0.3s ease-out;
  position: relative;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.msg.user {
  background: var(--user-bg);
  align-self: flex-end;
  margin-left: auto;
  border: 1px solid var(--border);
  border-bottom-right-radius: 4px;
}

.msg.agent {
  background: var(--agent-bg);
  align-self: flex-start;
  border: 1px solid var(--border);
  border-bottom-left-radius: 4px;
}

.msg.error {
  background: rgba(239, 68, 68, 0.1);
  border-color: rgba(239, 68, 68, 0.3);
  color: var(--error);
}

.msg .role-badge {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.msg .role-badge::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
}

.msg.user .role-badge::before {
  background: var(--accent-hover);
}

/* Loading indicator */
.loading {
  opacity: 0.7;
  font-style: italic;
  color: var(--text-secondary);
}

.loading::after {
  content: '';
  animation: dots 1.5s infinite;
}

@keyframes dots {
  0%, 20% { content: '.'; }
  40% { content: '..'; }
  60%, 100% { content: '...'; }
}

/* Input area */
#input-area {
  padding: 16px 24px;
  background: var(--bg-secondary);
  border-top: 1px solid var(--border);
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

.input-wrapper {
  flex: 1;
  position: relative;
}

#user-input {
  width: 100%;
  padding: 14px 18px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font-size: 14px;
  font-family: 'Inter', sans-serif;
  resize: none;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  min-height: 48px;
  max-height: 200px;
}

#user-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}

#user-input::placeholder {
  color: var(--text-muted);
}

#send-btn {
  padding: 14px 24px;
  border: none;
  border-radius: var(--radius);
  background: linear-gradient(135deg, var(--accent), #8b5cf6);
  color: white;
  font-size: 14px;
  font-weight: 500;
  font-family: 'Inter', sans-serif;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 8px;
  white-space: nowrap;
}

#send-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px var(--accent-glow);
}

#send-btn:active {
  transform: translateY(0);
}

#send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

#send-btn svg {
  width: 16px;
  height: 16px;
}

/* Welcome screen */
.welcome {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 40px;
  gap: 16px;
}

.welcome-icon {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, var(--accent), #8b5cf6);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  margin-bottom: 8px;
  box-shadow: 0 8px 32px var(--accent-glow);
}

.welcome h1 {
  font-size: 24px;
  font-weight: 600;
  color: var(--text-primary);
}

.welcome p {
  font-size: 14px;
  color: var(--text-secondary);
  max-width: 400px;
  line-height: 1.6;
}

.welcome-suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
  justify-content: center;
}

.suggestion {
  padding: 10px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.2s;
}

.suggestion:hover {
  border-color: var(--accent);
  color: var(--text-primary);
  background: var(--bg-tertiary);
}

/* Responsive */
@media (max-width: 640px) {
  .msg { max-width: 90%; }
  .header { padding: 12px 16px; }
  #chat { padding: 16px; }
  #input-area { padding: 12px 16px; }
}
</style>
</head>
<body>
<div class="header">
  <div class="header-logo">A</div>
  <div class="header-title">Agente Fuse</div>
  <div class="header-subtitle">FreeLLMAPI · auto:smart</div>
</div>

<div id="chat">
  <div class="welcome" id="welcome">
    <div class="welcome-icon">🤖</div>
    <h1>Bienvenido a Agente Fuse</h1>
    <p>Asistente inteligente con acceso a herramientas de Git, búsqueda de archivos y más. Powered by FreeLLMAPI con 4 proveedores gratuitos.</p>
    <div class="welcome-suggestions">
      <div class="suggestion" onclick="usarSugerencia('¿Qué herramientas tienes disponibles?')">🔧 Ver herramientas</div>
      <div class="suggestion" onclick="usarSugerencia('Explícame cómo funciona este agente')">📖 Cómo funciona</div>
      <div class="suggestion" onclick="usarSugerencia('Hazme un favor pequeño')">✨ Ayúdame</div>
    </div>
  </div>
</div>

<div id="input-area">
  <div class="input-wrapper">
    <textarea id="user-input" placeholder="Escribí tu mensaje..." autofocus rows="1"></textarea>
  </div>
  <button id="send-btn" onclick="enviar()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
    </svg>
    Enviar
  </button>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const welcome = document.getElementById('welcome');

// Auto-resize textarea
input.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

// Enter para enviar, Shift+Enter para nueva línea
input.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    enviar();
  }
});

function usarSugerencia(texto) {
  input.value = texto;
  input.focus();
  enviar();
}

function addMsg(role, text) {
  if (welcome) welcome.remove();
  
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  
  const badge = document.createElement('div');
  badge.className = 'role-badge';
  badge.textContent = role === 'user' ? 'Tú' : 'Agente';
  div.appendChild(badge);
  
  const content = document.createElement('div');
  content.textContent = text;
  div.appendChild(content);
  
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function enviar() {
  const text = input.value.trim();
  if (!text) return;
  
  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;
  
  addMsg('user', text);
  
  const loading = document.createElement('div');
  loading.className = 'msg agent loading';
  const badge = document.createElement('div');
  badge.className = 'role-badge';
  badge.textContent = 'Agente';
  loading.appendChild(badge);
  const content = document.createElement('div');
  content.textContent = 'Pensando';
  loading.appendChild(content);
  chat.appendChild(loading);
  chat.scrollTop = chat.scrollHeight;
  
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: text})
    });
    const data = await res.json();
    loading.remove();
    
    if (data.response) {
      addMsg('agent', data.response);
    } else if (data.error) {
      addMsg('error', data.error);
    }
  } catch(e) {
    loading.remove();
    addMsg('error', 'No se pudo conectar al servidor');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

input.focus();
</script>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _json_response(self, data, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass  # cliente cerró la conexión

    def _not_found(self):
        self._json_response({"error": "no encontrado"}, 404)

    def log_message(self, format, *args):
        logger.info(f"HTTP {args[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(port=8080):
    print(f"Iniciando servidor en http://localhost:{port}")
    print("  Abre http://localhost:{port} en tu navegador")
    print("  POST /chat para usar via API")
    print("  Ctrl+C para detener")
    server = HTTPServer(("0.0.0.0", port), AgenteHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
        server.shutdown()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run(port)
