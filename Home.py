# Home.py — Chat UI + Sidebar (logo → nav → search → tools), Light theme with TomTom red (#D30000)

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

# ---------- LLM Settings ----------
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

# Minimal CSS, add TomTom red (#D30000) styling
st.markdown(
    """
    <style>
      /* Hide default "Pages" nav */
      [data-testid="stSidebarNav"], section[data-testid="stSidebar"] nav { display: none !important; }

      /* Hide default chat avatars */
      [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] { display: none !important; }

      /* Chat margins */
      .chat-margin-container { margin-left: 100px; margin-right: 100px; }
      @media (max-width: 900px) {
        .chat-margin-container { margin-left: 10px; margin-right: 10px; }
      }

      /* --- TomTom red theme additions --- */
      /* Sidebar section titles */
      .css-1d391kg, .css-hby737, h3, h4 {
        color: #D30000 !important;
      }

      /* Buttons */
      .stButton>button {
        background-color: #D30000 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 9999px !important;
      }
      .stButton>button:hover {
        filter: brightness(1.1);
      }

      /* Chat header underline */
      hr {
        border: none;
        border-top: 2px solid #D30000 !important;
        margin: 0.5em 0;
      }

      /* Scrollbar (optional branding) */
      ::-webkit-scrollbar-thumb {
        background: #D30000 !important;
        border-radius: 4px;
      }
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
# ❖ Country list & helpers |
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
# ❖ Prompt Flow helpers / LLM (shortened for clarity) |
# ==========================
def to_pf_chat_history(msgs: List[Dict[str, str]], max_pairs: int = 6) -> List[Dict]:
    pairs, cur_user = [], None
    for m in msgs:
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if not text: continue
        if role == "user": cur_user = text
        elif role == "assistant" and cur_user is not None:
            pairs.append({"inputs": {"chat_input": cur_user}, "outputs": {"chat_output": text}})
            cur_user = None
    return pairs[-max_pairs:]

def get_llm_response(prompt: str) -> str:
    if not LLM_API_KEY or not LLM_ENDPOINT:
        return "⚠️ Missing LLM config"
    headers = {"Content-Type": "application/json"}
    if ".inference.ml.azure.com" in LLM_ENDPOINT.lower():
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    else:
        headers["api-key"] = LLM_API_KEY
    history_pf = to_pf_chat_history(st.session_state.get(SK_MSGS, [])) or \
                 [{"inputs": {"chat_input": "Hi"}, "outputs": {"chat_output": "Hello!"}}]
    payload = {"inputs": {"chat_input": prompt, "chat_history": history_pf}}
    try:
        resp = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return (data.get("outputs") or {}).get("chat_output") or data.get("chat_output") or str(data)
        return f"⚠️ Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"⚠️ Network error: {e}"

# ==========================
# ❖ Markdown/HTML helpers  |
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
        show_logo(center=True, max_width="180px")

        st.markdown("### Navigation")
        try:
            st.page_link("Home.py", label="🏠 Chat", icon=None)
        except Exception:
            st.write("• Chat")

        st.markdown("---")

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
                st.query_params.clear(); st.query_params["country"] = iso
                st.session_state["__go_country__"] = True
                st.rerun()

        st.markdown("---")

        # 🔴 Clear Conversation button (TomTom red via CSS)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state[SK_MSGS] = [
                {"role": "assistant", "content": "Hi! I am an AI model trained on RSM Data. How can I help today?"}
            ]
            st.rerun()

# ==========================
# ❖ Top-level redirect guard
# ==========================
if st.session_state.get("__go_country__") or st.query_params.get("country"):
    if "__go_country__" in st.session_state: del st.session_state["__go_country__"]
    try:
        st.switch_page("pages/_Country.py")  # hidden file
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
        with st.spinner("Thinking…"):
            reply = get_llm_response(prompt.strip())
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
