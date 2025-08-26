# Home.py — Chat UI + Sidebar (logo → nav → search → tools), Light theme + TomTom red
# Robust Prompt Flow calling with strict filter for AML default "compute instance" sample output.

from __future__ import annotations
import os, json, base64, html, re, requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st

# =======================
# ❖ Config / Constants  |
# =======================
APP_TITLE = "TomTom Tax Agent"
APP_ICON = ".streamlit/TomTom-Logo.png"
LOGO_PATH = Path(".streamlit/TomTom-Logo.png")

LLM_API_KEY = os.getenv("AZURE_API_KEY", "")
LLM_ENDPOINT = os.getenv("AZURE_API_ENDPOINT", "")

SK_MSGS = "messages"
MAX_CONTEXT_MESSAGES = 12

# =========================
# ❖ Page / Setup          |
# =========================
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

if PIL_AVAILABLE and Path(APP_ICON).exists():
    try:
        _icon_obj = Image.open(APP_ICON)
    except Exception:
        _icon_obj = APP_ICON
else:
    _icon_obj = APP_ICON

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=_icon_obj,
    initial_sidebar_state="expanded",
)

# Minimal CSS (keep light theme from .streamlit/config.toml; add TomTom red accents)
st.markdown(
    """
    <style>
      /* Hide Streamlit's built-in Pages nav so our logo sits at the very top */
      [data-testid="stSidebarNav"], section[data-testid="stSidebar"] nav { display: none !important; }

      /* Optional: hide default chat avatars (we render our own emoji) */
      [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] { display: none !important; }

      /* Chat margins */
      .chat-margin-container { margin-left: 100px; margin-right: 100px; }
      @media (max-width: 900px) {
        .chat-margin-container { margin-left: 10px; margin-right: 10px; }
      }

      /* TomTom red accents */
      .stButton>button {
        background-color: #D30000 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 9999px !important;
      }
      .stButton>button:hover { filter: brightness(1.1); }
      h3, h4 { color: #D30000 !important; }
      hr { border: none; border-top: 2px solid #D30000 !important; margin: .5rem 0; }
      ::-webkit-scrollbar-thumb { background: #D30000 !important; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def logo_img_base64() -> Optional[str]:
    if LOGO_PATH and LOGO_PATH.exists():
        try:
            return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        except Exception:
            return None
    return None

def show_logo(center: bool = True, max_width: str = "180px") -> None:
    b64 = logo_img_base64()
    if b64:
        align = "margin-left:auto;margin-right:auto;" if center else ""
        st.markdown(
            f"""
            <div style="text-align:center; margin: 0 0 .25rem 0;">
                <img src="data:image/png;base64,{b64}" style="max-width:{max_width}; {align}" />
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================
# ❖ Countries              |
# ==========================
COUNTRIES: List[Tuple[str, str]] = [
    ("Netherlands", "NL"), ("Belgium", "BE"), ("Germany", "DE"), ("France", "FR"),
    ("United Kingdom", "GB"), ("Spain", "ES"), ("Italy", "IT"), ("Poland", "PL"),
    ("United States", "US"), ("Canada", "CA"),
]
def find_country_by_name(name: str) -> Optional[Tuple[str, str]]:
    for n, c in COUNTRIES:
        if n == name:
            return (n, c)
    return None

# ==========================
# ❖ Prompt Flow helpers    |
# ==========================
def to_pf_chat_history(msgs: List[Dict[str, str]], max_pairs: int = 6) -> List[Dict]:
    """Convert Streamlit chat history to Prompt Flow pairs."""
    pairs, cur_user = [], None
    for m in msgs:
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if role == "user":
            cur_user = text
        elif role == "assistant" and cur_user is not None:
            pairs.append({"inputs": {"chat_input": cur_user},
                          "outputs": {"chat_output": text}})
            cur_user = None
    if not pairs:
        pairs = [{
            "inputs": {"chat_input": "Hi"},
            "outputs": {"chat_output": "Hello! How can I assist you today?"}
        }]
    return pairs[-max_pairs:]

# --- Strict detector for AML tutorial default output ---
DEFAULT_SIGNATURES = [
    "ml_client.compute.begin_create_or_update(compute_instance)",
    "from azure.ai.ml import mlclient",
    "from azure.ai.ml.entities import computeinstance",
    "from azure.identity import defaultazurecredential",
    "standard_ds3_v2",
    "steps to create a compute instance",
    "to create an azure machine learning compute instance",
    "pip install azure-ai-ml",
    "computeinstance(",
    "begin_create_or_update(",
    "compute instance '",
    "azure machine learning workspace",
    "vm size (e.g., standard_ds3_v2)",
    "monitor the creation process",
]
def looks_like_default(text: str) -> bool:
    """Return True if text smells like the AML 'create compute instance' tutorial."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(sig in t for sig in DEFAULT_SIGNATURES):
        return True
    # Common combos/tutorial markers
    if ("compute instance" in t and ("azureml" in t or "azure.ai.ml" in t)):
        return True
    if "###" in t and "```" in t and "pip install azure-ai-ml" in t:
        return True
    return False

# ==========================
# ❖ LLM Call               |
# ==========================
def get_llm_response(prompt: str, context: str) -> str:
    """
    Robust caller for a Prompt Flow scoring endpoint that expects:
      inputs.chat_input (string) + inputs.chat_history (list of PF pairs)
    Tries:
      - URL as-is and with '/score'
      - Bearer vs api-key auth
      - Foundry-style vs AML-style request bodies
    Filters AML's sample 'compute instance' blob in ALL code paths.
    """
    if not LLM_API_KEY or not LLM_ENDPOINT:
        return "⚠️ Missing LLM configuration. Set AZURE_API_KEY and AZURE_API_ENDPOINT."

    endpoint = LLM_ENDPOINT.rstrip("/")
    urls = [endpoint] + ([endpoint + "/score"] if not endpoint.endswith("/score") else [])

    def bearer_headers():
        return {"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"}
    def apikey_headers():
        return {"Content-Type": "application/json", "api-key": LLM_API_KEY}

    headers_list = [bearer_headers(), apikey_headers()]  # try both

    history_pf = to_pf_chat_history(st.session_state.get(SK_MSGS, []))

    body_foundry = {"inputs": {"chat_input": prompt, "chat_history": history_pf}}
    body_aml = {"input_data": {"inputs": {"chat_input": prompt, "chat_history": history_pf}}}
    payloads = [("foundry.inputs", body_foundry), ("aml.input_data.inputs", body_aml), ("raw.top", {"chat_input": prompt, "chat_history": history_pf})]

    last_status = last_text = last_where = None

    for url in urls:
        for headers in headers_list:
            for tag, body in payloads:
                where = f"{url} [{ 'Bearer' if 'Authorization' in headers else 'api-key' } | {tag}]"
                try:
                    resp = requests.post(url, headers=headers, json=body, timeout=90)
                    last_status, last_text, last_where = resp.status_code, resp.text, where
                    if resp.status_code != 200:
                        continue

                    # Try JSON first
                    try:
                        data = resp.json()
                    except Exception:
                        # Non-JSON text
                        txt = (resp.text or "").strip()
                        if txt and looks_like_default(txt):
                            continue
                        if txt:
                            return txt
                        continue

                    # Extract content from common shapes
                    content = (
                        (data.get("outputs") or {}).get("chat_output")
                        or data.get("chat_output")
                        or data.get("output")
                        or data.get("prediction")
                        or data.get("result")
                        or data.get("value")
                    )
                    if content:
                        if looks_like_default(content):
                            continue
                        return content

                    # If no obvious field, stringify briefly for debug — but block default text
                    cand = json.dumps(data, ensure_ascii=False)
                    if cand and looks_like_default(cand):
                        continue
                    if cand:
                        return cand[:4000]

                except requests.RequestException as e:
                    last_text = f"Network error: {e}"
                    continue

    # All attempts failed or were blocked as default
    debug = f"Last attempt: {last_where} → HTTP {last_status}\nBody (first 400 chars):\n{(last_text or '')[:400]}"
    return (
        "⚠️ The endpoint responded with a sample/default output or an error. "
        "This usually means the request schema or auth header doesn't match the deployed flow. "
        "Open the endpoint's **Test** tab, run a test, click **View request**, and align the JSON.\n\n" + debug
    )

# ==========================
# ❖ Simple render helpers  |
# ==========================
def format_llm_reply_to_html(raw_text: str) -> str:
    return "<p>" + html.escape((raw_text or "").strip()).replace("\n", "<br>") + "</p>"

def render_chat_history(messages: List[Dict[str, str]]) -> None:
    assistant_icon = "🤖"; user_icon = "🕵️‍♂️"
    for m in messages:
        role = m.get("role", "assistant")
        content = (m.get("content", "") or "").strip()
        chat_role = "user" if role == "user" else "assistant"
        inner_html = html.escape(content).replace("\n", "<br>") if chat_role == "user" else format_llm_reply_to_html(content)
        icon_html = f'<div style="font-size:32px;margin-right:12px;">{user_icon if chat_role=="user" else assistant_icon}</div>'
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;margin-bottom:1.2em;">
              {icon_html}
              <div style="max-width:85%;font-size:1.04em;">{inner_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================
# ❖ Sidebar (logo → nav → search → tools)
# ==========================
def render_sidebar_home() -> None:
    with st.sidebar:
        # 1) Logo
        show_logo(center=True, max_width="180px")

        # 2) Navigation
        st.markdown("### Navigation")
        try:
            st.page_link("Home.py", label="🏠 Chat", icon=None)
        except Exception:
            st.write("• Chat")
        st.markdown("---")

        # 3) Country search
        st.markdown("#### Country search")
        country_name = st.selectbox(
            "Search or select a country",
            options=[n for n, _ in COUNTRIES],
            index=None,
            placeholder="Type to search…",
        )
        if country_name:
            selected = find_country_by_name(country_name)
            if selected:
                n, iso = selected
                st.session_state["selected_country_name"] = n
                st.session_state["selected_country_iso"] = iso
                st.query_params.clear()
                st.query_params["country"] = iso
                st.session_state["__go_country__"] = True
                st.rerun()

        st.markdown("---")

        # 4) Tools — Clear conversation (TomTom red via CSS above)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state[SK_MSGS] = [
                {"role": "assistant", "content": "Hi! I am an AI model trained on RSM Data. How can I help today?"}
            ]
            st.rerun()

# ==========================
# ❖ Top-level redirect guard
# ==========================
if st.session_state.get("__go_country__") or st.query_params.get("country"):
    if "__go_country__" in st.session_state:
        del st.session_state["__go_country__"]
    try:
        st.switch_page("pages/_Country.py")
    except Exception:
        st.rerun()

# ==========================
# ❖ UI: Chat               |
# ==========================
def chat_ui() -> None:
    render_sidebar_home()

    st.title(APP_TITLE)
    st.markdown("---")
    st.header("📖 Chat Assistant")

    st.markdown('<div class="chat-margin-container">', unsafe_allow_html=True)

    st.session_state.setdefault(
        SK_MSGS,
        [{"role": "assistant", "content": "Hi! I am an AI model trained on RSM Data. How can I help today?"}],
    )

    render_chat_history(st.session_state[SK_MSGS])

    prompt = st.chat_input("Type your message and press enter", key="chat_prompt")
    if prompt and prompt.strip():
        st.session_state[SK_MSGS].append({"role": "user", "content": prompt.strip()})
        recent = st.session_state[SK_MSGS][-MAX_CONTEXT_MESSAGES:]
        context_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

        with st.spinner("Thinking…"):
            reply = get_llm_response(prompt.strip(), context_text)

        st.session_state[SK_MSGS].append({"role": "assistant", "content": reply})
        if len(st.session_state[SK_MSGS]) > MAX_CONTEXT_MESSAGES:
            st.session_state[SK_MSGS] = st.session_state[SK_MSGS][-MAX_CONTEXT_MESSAGES:]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# ❖ App Entry              |
# ==========================
def main() -> None:
    chat_ui()

if __name__ == "__main__":
    main()
