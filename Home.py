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
    """
    Calls your endpoint robustly:
    - Azure OpenAI (chat/completions): builds proper messages payload
    - Azure AI / Prompt Flow / AML endpoints: tries multiple schemas
    - Filters out known AML default sample text so it never reaches the user
    """
    import json, requests

    # --- sanity checks ---
    if not LLM_API_KEY or not LLM_ENDPOINT:
        return "⚠️ Missing LLM configuration. Please set AZURE_API_KEY and AZURE_API_ENDPOINT."

    endpoint_lower = LLM_ENDPOINT.lower()
    is_aml = ".inference.ml.azure.com" in endpoint_lower
    is_foundry = (".inference.azureai.io" in endpoint_lower) or ("ai.azure.com" in endpoint_lower)
    is_aoai = ("openai.azure.com" in endpoint_lower) or ("/chat/completions" in endpoint_lower)

    # --- headers ---
    headers = {"Content-Type": "application/json"}
    if is_aml:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    else:
        headers["api-key"] = LLM_API_KEY

    # --- build a lightweight chat history (Prompt Flow style pairs already exist) ---
    def pf_history():
        hist = to_pf_chat_history(st.session_state.get(SK_MSGS, []))
        return hist or [{"inputs": {"chat_input": "Hi"}, "outputs": {"chat_output": "Hello!"}}]

    # --- helper: detect AML default sample output & suppress it ---
    DEFAULT_SIGNATURES = [
        "ml_client.compute.begin_create_or_update(compute_instance)",
        "from azure.ai.ml import mlclient",
        "from azure.ai.ml.entities import computeinstance",
        "DefaultAzureCredential",
        "Standard_DS3_v2",
        "steps to create a compute instance",
        "stopping a compute instance",
    ]
    def looks_like_default(text: str) -> bool:
        t = (text or "").lower()
        if len(t) < 60:
            return False
        hits = sum(sig.lower() in t for sig in DEFAULT_SIGNATURES)
        aml_combo = (("compute instance" in t) and ("azureml" in t or "azure.ai.ml" in t))
        return hits >= 2 or aml_combo

    # --- 1) Azure OpenAI chat/completions path ---
    if is_aoai:
        # Build OpenAI-style messages from your session history
        messages = []
        # Optional system prompt
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        for m in st.session_state.get(SK_MSGS, []):
            role = m.get("role", "assistant")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in ("user", "assistant", "system"):
                role = "user" if role == "human" else "assistant"
            messages.append({"role": role, "content": content})
        # Add the new user prompt
        messages.append({"role": "user", "content": prompt})

        try:
            resp = requests.post(LLM_ENDPOINT, headers=headers, json={"messages": messages, "temperature": 0.2}, timeout=90)
            if resp.status_code != 200:
                return f"⚠️ AOAI error {resp.status_code}: {resp.text[:500]}"
            data = resp.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            return content or "⚠️ AOAI returned no content."
        except Exception as e:
            return f"⚠️ AOAI request failed: {e}"

    # --- 2) Prompt Flow / AML schemas (try several) ---
    payloads = [
        # Foundry / AOAI serverless style (Prompt Flow)
        ("foundry.inputs", {"inputs": {"chat_input": prompt, "chat_history": pf_history()}}),
        # AML managed online endpoint with input_data.inputs
        ("aml.input_data.inputs", {"input_data": {"inputs": {"chat_input": prompt, "chat_history": pf_history()}}}),
        # Raw top-level fields some runtimes accept
        ("raw.top", {"chat_input": prompt, "chat_history": pf_history()}),
        # Some flows accept a simple input_string
        ("aml.input_data.input_string", {"input_data": {"input_string": prompt}}),
        # Some flows accept bare "input_string"
        ("raw.input_string", {"input_string": prompt}),
    ]

    last_status = None
    last_text = None
    last_tag = None

    for tag, body in payloads:
        try:
            resp = requests.post(LLM_ENDPOINT, headers=headers, json=body, timeout=90)
            last_status, last_text, last_tag = resp.status_code, resp.text, tag
            if resp.status_code != 200:
                continue

            # Try JSON first
            try:
                data = resp.json()
                # common shapes
                content = (
                    (data.get("outputs") or {}).get("chat_output")
                    or data.get("chat_output")
                    or data.get("output")
                    or data.get("prediction")
                    or data.get("result")
                    or data.get("value")
                )
                if not content:
                    # stringify the whole object if no obvious field
                    cand = json.dumps(data, ensure_ascii=False)[:4000]
                    if cand and not looks_like_default(cand):
                        return cand
                    else:
                        continue

                # filter sample/default
                if looks_like_default(content):
                    continue

                return content

            except ValueError:
                # Non-JSON response; use text
                txt = (resp.text or "").strip()
                if txt and not looks_like_default(txt):
                    return txt
                else:
                    continue

        except requests.RequestException as e:
            last_text = f"Network error calling LLM: {e}"
            continue

    # --- If all schemas failed or looked like default ---
    debug = f"[{last_tag}] HTTP {last_status}; First 300 chars:\n{(last_text or '')[:300]}"
    return (
        "⚠️ Your endpoint returned a default/sample response, which usually means the request body "
        "schema didn’t match. Open your endpoint’s **Test** tab and copy the exact working request body, "
        "then update `get_llm_response` to use that schema.\n\n"
        + debug
    )

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
