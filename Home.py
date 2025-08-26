# app.py — Chat UI with Country search (routes to Country page)

from __future__ import annotations
import os
import json
import base64
import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import requests
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
DISABLE_DEFAULT_FILTER = False

# =========================
# ❖ Page / Global Styling |
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

def get_assistant_icon_b64() -> str:
    return "🤖"

def get_user_icon_b64() -> str:
    return "🕵️‍♂️"

def inject_light_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --primary-color: #D30000;
                --background-color: #ffffff;
                --app-bg: #ffffff;
                --secondary-background-color: #f5f7fb;
                --text-color:#0f172a;
                --link-color: #0B5FFF;
                --border-color: #e5e7eb;
                --code-bg: #f3f4f6;
                --base-radius: 0.5rem;
                --button-radius: 9999px;
            }
            html, body, .stApp, [class*="css"] {
                background: var(--background-color) !important;
                color: var(--text-color) !important;
            }
            section[data-testid="stSidebar"] {
                background: var(--secondary-background-color) !important;
                border-right: 1px solid var(--border-color) !important;
            }
            a { color: var(--link-color) !important; }
            pre, code, kbd, samp { background: var(--code-bg) !important; color: var(--text-color) !important; }
            .stButton>button {
                background: var(--primary-color) !important;
                color: #fff !important;
                border: none !important;
                border-radius: var(--button-radius) !important;
            }
            .stButton>button:hover { filter: brightness(1.05); }
            .sidebar-section-title {
                font-size: 0.9rem;
                letter-spacing: .02em;
                color: #334155;
                text-transform: uppercase;
                margin: .5rem 0 .25rem 0;
            }
            [data-testid="chatAvatarIcon-user"], 
            [data-testid="chatAvatarIcon-assistant"] {
                display: none !important;
            }
            .chat-margin-container { margin-left: 100px !important; margin-right: 100px !important; }
            @media (max-width: 900px) {
                .chat-margin-container { margin-left: 10px !important; margin-right: 10px !important; }
            }
            .chat-margin-container div[style*="box-shadow"] { box-shadow: none !important; }
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

def show_logo(center: bool = True) -> None:
    b64 = logo_img_base64()
    if b64:
        align = "margin-left:auto;margin-right:auto;" if center else ""
        st.markdown(
            f'<img class="brand-logo" src="data:image/png;base64,{b64}" style="max-width: 180px; {align}" />',
            unsafe_allow_html=True,
        )

# ==========================
# ❖ Country search helpers |
# ==========================
# Minimal list (extend as needed)
COUNTRIES: List[Tuple[str, str]] = [
    ("Netherlands", "NL"),
    ("Belgium", "BE"),
    ("Germany", "DE"),
    ("France", "FR"),
    ("United Kingdom", "GB"),
    ("Spain", "ES"),
    ("Italy", "IT"),
    ("Poland", "PL"),
    ("United States", "US"),
    ("Canada", "CA"),
]

def iso_to_flag(iso2: str) -> str:
    """Convert ISO-3166 alpha-2 to emoji flag."""
    iso = (iso2 or "").upper()
    if len(iso) != 2 or not iso.isalpha():
        return "🏳️"
    return chr(ord(iso[0]) - 65 + 0x1F1E6) + chr(ord(iso[1]) - 65 + 0x1F1E6)

def find_country_by_name(name: str) -> Optional[Tuple[str, str]]:
    for n, c in COUNTRIES:
        if n == name:
            return (n, c)
    return None

# ==========================
# ❖ Prompt Flow helpers    |
# ==========================
def to_pf_chat_history(msgs: List[Dict[str, str]], max_pairs: int = 6) -> List[Dict]:
    pairs = []
    cur_user = None
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
    return pairs[-max_pairs:]

DEFAULT_SIGNATURES = [
    "steps to create a compute instance",
    "install the azureml sdk v2",
    "from azure.ai.ml import mlclient",
    "from azure.ai.ml.entities import computeinstance",
    "from azure.identity import defaultazurecredential",
    "ml_client.compute.begin_create_or_update(compute_instance)",
    "standard_ds3_v2",
    "stopping a compute instance",
]

def looks_like_default(text: str) -> bool:
    if DISABLE_DEFAULT_FILTER:
        return False
    t = (text or "").strip().lower()
    if len(t) < 120:
        return False
    hits = sum(sig in t for sig in DEFAULT_SIGNATURES)
    aml_combo = (
        ("compute instance" in t and "azureml sdk" in t) or
        ("compute instance" in t and "ml_client" in t) or
        ("azure.ai.ml" in t and "computeinstance" in t)
    )
    return hits >= 2 or aml_combo

# ==========================
# ❖ LLM Call               |
# ==========================
def get_llm_response(prompt: str) -> str:
    if not LLM_API_KEY or not LLM_ENDPOINT:
        raise RuntimeError("Missing LLM configuration. Set AZURE_API_KEY and AZURE_API_ENDPOINT.")

    endpoint_lower = LLM_ENDPOINT.lower()
    is_aml = ".inference.ml.azure.com" in endpoint_lower
    is_foundry = (".inference.azureai.io" in endpoint_lower) or ("ai.azure.com" in endpoint_lower)

    headers = {"Content-Type": "application/json"}
    if is_aml:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    else:
        headers["api-key"] = LLM_API_KEY

    history_pf = to_pf_chat_history(st.session_state.get(SK_MSGS, []))
    if not history_pf:
        history_pf = [{"inputs": {"chat_input": "Hi"}, "outputs": {"chat_output": "Hello!"}}]

    payloads = [
        ("foundry.inputs", {"inputs": {"chat_input": prompt, "chat_history": history_pf}}),
        ("aml.input_data.inputs", {"input_data": {"inputs": {"chat_input": prompt, "chat_history": history_pf}}}),
        ("raw.top", {"chat_input": prompt, "chat_history": history_pf}),
    ]

    last_status = None
    last_text = None
    last_tag = None

    for tag, body in payloads:
        try:
            resp = requests.post(LLM_ENDPOINT, headers=headers, json=body, timeout=90)
            last_status = resp.status_code
            last_text = resp.text
            last_tag = tag

            if resp.status_code != 200:
                continue

            try:
                data = resp.json()
                content = (
                    (data.get("outputs") or {}).get("chat_output")
                    or data.get("chat_output")
                    or data.get("output")
                    or data.get("prediction")
                    or data.get("result")
                    or data.get("value")
                )
                if not content:
                    cand = json.dumps(data, ensure_ascii=False)
                    if cand and not looks_like_default(cand):
                        return cand[:4000]
                    continue
                if looks_like_default(content):
                    continue
                return content
            except Exception:
                txt = (resp.text or "").strip()
                if txt and not looks_like_default(txt):
                    return txt
                else:
                    continue

        except requests.RequestException as e:
            last_text = f"Network error calling LLM: {e}"
            continue

    debug = f"[{last_tag}] HTTP {last_status}; Body (first 800 chars): {str(last_text)[:800]}"
    raise RuntimeError(
        "The endpoint responded but appears to return the flow's default sample output "
        "(schema mismatch). Try another request schema or copy the exact Test tab body.\n\n" + debug
    )

# ==========================
# ❖ Markdown/HTML helpers  |
# ==========================
def _tidy_llm_text(text: str) -> str:
    t = (text or "").strip().replace("\r\n", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"^\s*\*\*TL;DR:\*\*\s*$", "### TL;DR", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"^\s*[•–*]\s+", "- ", t, flags=re.MULTILINE)
    return t

def _md_to_html_basic(md: str) -> str:
    md = _tidy_llm_text(md)
    lines = md.split("\n")
    out = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def fmt_inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', s)
        return s

    for raw in lines:
        ln = raw.rstrip()
        if re.match(r"^\s*---+\s*$", ln):
            close_lists(); out.append("<hr/>"); continue
        m = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", ln)
        if m:
            close_lists(); level = len(m.group(1))
            out.append(f"<h{level}>{fmt_inline(m.group(2))}</h{level}>"); continue
        m = re.match(r"^\s*\d+\.\s+(.+)$", ln)
        if m:
            if in_ul: out.append("</ul>"); in_ul = False
            if not in_ol: out.append("<ol>"); in_ol = True
            out.append(f"<li>{fmt_inline(m.group(1))}</li>"); continue
        m = re.match(r"^\s*-\s+(.+)$", ln)
        if m:
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append(f"<li>{fmt_inline(m.group(1))}</li>"); continue
        if not ln.strip():
            close_lists(); out.append(""); continue
        close_lists(); out.append(f"<p>{fmt_inline(ln)}</p>")

    close_lists()
    return "\n".join(out)

def format_llm_reply_to_html(raw_text: str) -> str:
    return _md_to_html_basic(raw_text)

def render_chat_history(messages: List[Dict[str, str]]) -> None:
    assistant_icon_b64 = get_assistant_icon_b64()
    user_icon_b64 = get_user_icon_b64()
    for m in messages:
        role = m.get("role", "assistant")
        content = (m.get("content", "") or "").strip()
        chat_role = "user" if role == "user" else "assistant"

        if chat_role == "assistant":
            inner_html = format_llm_reply_to_html(content)
            icon_html = f'<div style="font-size:32px;margin-right:12px;display:block;">{assistant_icon_b64}</div>'
        else:
            inner_html = html.escape(content).replace("\n", "<br>")
            icon_html = f'<div style="font-size:32px;margin-right:12px;display:block;">{user_icon_b64}</div>'

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;width:100%;margin-bottom:1.2em;">
              {icon_html}
              <div style="
                  background:none;color:var(--text-color);border-radius:0;padding:0.2em 1.1em;
                  box-shadow:none;max-width:85%;line-height:1.5;word-wrap:break-word;white-space:pre-wrap;
                  font-size:1.04em;min-height:32px;text-align:left;display:block;">
                {inner_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==========================
# ❖ UI: Chat               |
# ==========================
def chat_ui() -> None:
    # ---- SIDEBAR ----
    with st.sidebar:
        show_logo(center=True)
        st.markdown("### Country search")
        country_name = st.selectbox(
            "Search or select a country",
            options=[n for n, _ in COUNTRIES],
            index=None,
            placeholder="Type to search…",
        )
        # inside chat_ui(), after user picks `country_name`
        if country_name:
            selected = find_country_by_name(country_name)
            if selected:
                n, iso = selected
                    # 1) persist in session (works across pages)
                st.session_state["selected_country_name"] = n
                st.session_state["selected_country_iso"] = iso

        # 2) write URL param using the NEW API
                st.query_params.clear()            # ensure a clean slate
                st.query_params["country"] = iso   # e.g. "NL"

        # 3) navigate to the page
                st.switch_page("pages/_Country.py")


        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state[SK_MSGS] = [
                {"role": "assistant", "content": "Hi! I am an AI model trained on RSM Data. How can I help today?"}
            ]
            st.rerun()

    # ---- MAIN CONTENT
    st.title(APP_TITLE)
    st.markdown("---")
    st.header("🗺️ Chat Assistant")

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
            try:
                reply = get_llm_response(prompt.strip())
            except Exception as e:
                reply = f"Sorry, I hit an error calling the model:\n\n```\n{e}\n```"
        st.session_state[SK_MSGS].append({"role": "assistant", "content": reply})
        if len(st.session_state[SK_MSGS]) > MAX_CONTEXT_MESSAGES:
            st.session_state[SK_MSGS] = st.session_state[SK_MSGS][-MAX_CONTEXT_MESSAGES:]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# ❖ App Entry              |
# ==========================
def main() -> None:
    inject_light_css()
    chat_ui()

if __name__ == "__main__":
    main()
