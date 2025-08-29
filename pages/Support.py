# pages/Support.py — Support form → Outlook Web compose (prepopulated)
# - Brand colours: #009CDE (primary), #3F9C35 (accent), #888B8D (muted)
# - Sidebar like Home + Back to Home
# - Company logo placeholder if no image found
# - Clean subject/body formatting

from __future__ import annotations
import os, base64
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlencode, quote
import streamlit as st

# ========= Config =========
APP_TITLE = "Support – TomTom Tax Agent"
APP_ICON  = ".streamlit/TomTom-Logo.png"
# Your company logo (put a PNG here to replace the placeholder)
COMPANY_LOGO_PATH = Path(".streamlit/RSM_Standard_Logo_RGB No Background.png")
TOMTOM_LOGO_PATH  = Path(".streamlit/TomTom-Logo.png")

# Pre-populated recipients (edit these)
SUPPORT_TO = os.getenv("SUPPORT_TO", "IZalinyan@rsmnl.nl")  # comma-separated
SUPPORT_CC = os.getenv("SUPPORT_CC", "HvLoenen@rsmnl.nl, MvdBroek@rsmnl.nl")  # comma-separated

# Categories & Countries
CATEGORIES: List[str] = [
    "Access / Permissions",
    "Data / Content",
    "Bug / Error",
    "Feature Request",
    "Billing / Licensing",
    "Other",
]
COUNTRIES: List[Tuple[str, str]] = [
    ("Netherlands", "NL"), ("Belgium", "BE"), ("Germany", "DE"), ("France", "FR"),
    ("United Kingdom", "GB"), ("Spain", "ES"), ("Italy", "IT"), ("Poland", "PL"),
    ("United States", "US"), ("Canada", "CA"), ("N/A", "N/A"),
]

def find_country_by_name(name: str) -> Optional[Tuple[str, str]]:
    for n, c in COUNTRIES:
        if n == name:
            return (n, c)
    return None

# ========= Page setup =========
try:
    from PIL import Image
    _icon_obj = Image.open(APP_ICON) if Path(APP_ICON).exists() else APP_ICON
except Exception:
    _icon_obj = APP_ICON

st.set_page_config(page_title=APP_TITLE, page_icon=_icon_obj, initial_sidebar_state="expanded")

# ======= Theming / CSS =======
st.markdown(
    """
    <style>
      :root {
        --primary: #009CDE;   /* blue */
        --accent:  #3F9C35;   /* green */
        --muted:   #888B8D;   /* grey */
      }

      /* Hide Streamlit's default page nav in sidebar */
      [data-testid="stSidebarNav"], section[data-testid="stSidebar"] nav { display:none !important; }

      /* Buttons (primary) */
      .stButton > button, .stLinkButton > a {
        background-color: var(--primary) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 9999px !important;
      }
      .stButton > button:hover, .stLinkButton > a:hover { filter: brightness(1.06); }

      /* Titles and accents */
      h1.app-title { color: var(--primary) !important; margin: 0 0 .25rem 0; }
      h3, h4 { color: var(--primary) !important; }
      .help-hint { color: var(--muted); font-size: 0.9rem; margin-top: -0.5rem; }
      a, .stMarkdown a { color: var(--accent) !important; }

      /* Sidebar headings/separators */
      .sidebar-section-title { color: var(--muted); text-transform: uppercase; font-size: 0.85rem; letter-spacing: .03em; }
      hr.sidebar-sep { border: none; border-top: 1px solid var(--muted); margin: .5rem 0; }

      /* Brand header */
      .brand-bar {
        display: flex; align-items: center; justify-content: space-between;
        gap: 1rem; padding: .5rem 0 .75rem 0; border-bottom: 1px solid var(--muted);
        margin-bottom: .5rem;
      }
      .brand-logo {
        max-height: 44px; border-radius: 6px;
      }
      .brand-placeholder {
        width: 180px; height: 44px; display:flex; align-items:center; justify-content:center;
        border: 1px dashed var(--muted); color: var(--muted); border-radius: 6px;
        font-size: 0.85rem;
      }

      .page-actions { display:flex; gap:.5rem; align-items:center; }

      /* Make text inputs a bit crisper with a border hint */
      .stTextInput > div > div > input, .stTextArea textarea, .stSelectbox > div > div {
        border: 1px solid rgba(0,0,0,0.1) !important;
        border-radius: 8px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========= Logo helpers =========
def _b64_from_path(p: Path) -> Optional[str]:
    try:
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        pass
    return None

def render_brand_bar() -> None:
    """Top brand bar: Company logo (placeholder if missing) + TomTom logo."""
    comp_b64 = _b64_from_path(COMPANY_LOGO_PATH)
    tt_b64   = _b64_from_path(TOMTOM_LOGO_PATH)

    left = f'<div class="brand-placeholder">Your Company Logo</div>' if not comp_b64 else \
           f'<img class="brand-logo" src="data:image/png;base64,{comp_b64}" alt="Company Logo"/>'
    right = "" if not tt_b64 else \
            f'<img class="brand-logo" src="data:image/png;base64,{tt_b64}" alt="TomTom Logo"/>'

    st.markdown(
        f"""
        <div class="brand-bar">
          <div>{left}</div>
          <div>{right}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========= Sidebar (logo + nav + country search) =========
def render_sidebar() -> None:
    with st.sidebar:

        # 2) Navigation
        st.markdown("### Navigation")
        try:
            st.page_link("Home.py", label="🏠 Chat", icon=None)
            st.page_link("pages/Support.py", label="❓ Support", icon=None)
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
                
# ========= Outlook Web compose helpers =========
def build_outlook_web_compose_url(
    to_emails: List[str],
    cc_emails: List[str],
    subject: str,
    body: str,
) -> str:
    """
    Outlook Web 'deeplink compose':
      https://outlook.office.com/mail/deeplink/compose?to=...&cc=...&subject=...&body=...
    """
    base = "https://outlook.office.com/mail/deeplink/compose"
    params = {
        "to": ",".join([e.strip() for e in to_emails if e.strip()]),
        "cc": ",".join([e.strip() for e in cc_emails if e.strip()]),
        "subject": subject or "",
        "body": body or "",
    }
    return f"{base}?{urlencode(params, doseq=False, safe=':/@')}"  # allow addresses and slashes in body

def build_mailto_fallback(
    to_emails: List[str],
    cc_emails: List[str],
    subject: str,
    body: str,
) -> str:
    """Fallback 'mailto:' for users without Outlook Web."""
    to_str = ",".join([e.strip() for e in to_emails if e.strip()])
    params = {
        "cc": ",".join([e.strip() for e in cc_emails if e.strip()]),
        "subject": subject or "",
        "body": body or "",
    }
    return f"mailto:{quote(to_str)}?{urlencode(params)}"

# ========= Page content =========
render_sidebar()
render_brand_bar()
st.markdown('<h1 class="app-title">Contact RSM</h1>', unsafe_allow_html=True)

# Back to Home button
cols = st.columns([1, 9])
with cols[0]:
    used_link = False
    try:
        st.page_link("Home.py", label="← Back to Home", icon=None)
        used_link = True
    except Exception:
        pass
    if not used_link:
        if st.button("← Back to Home"):
            try:
                st.switch_page("Home.py")
            except Exception:
                st.warning("Couldn’t switch pages automatically. Use the left sidebar or top navigation.")

st.write("Fill out the form and we’ll open a pre-filled email in **Outlook on the web** for you to review and send.")

with st.form("support_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        your_name  = st.text_input("Your name", "Illaih Westerhuis")
        category   = st.selectbox("Category", CATEGORIES, index=0)
    with col2:
        your_email   = st.text_input("Your email", "illaih.westerhuis@tomtom.com")
        country_name = st.selectbox("Country", [n for n, _ in COUNTRIES], index=0)

    short_summary = st.text_input("Short summary (subject line)", placeholder="Cannot access the Tax Agent dashboard")
    details = st.text_area(
        "Details (included in the email body)",
        height=220,
        placeholder=(
            "Describe the issue, steps to reproduce, expected results and any relevant links.\n"
            "Example:\n- Issue started today after logging in\n- Error message: ...\n- Impacted users: ...\n"
        ),
    )

    # Recipients (prepopulated but editable)
    to_default = [e.strip() for e in SUPPORT_TO.split(",") if e.strip()]
    cc_default = [e.strip() for e in SUPPORT_CC.split(",") if e.strip()]
    to_list = st.text_input("To", ", ".join(to_default))
    cc_list = st.text_input("CC", ", ".join(cc_default))

    submitted = st.form_submit_button("Submit & open Outlook Web")

if submitted:
    # Build subject/body
    iso = next((c for n, c in COUNTRIES if n == country_name), "")
    subject = f"[Support | {country_name or 'N/A'} | {category}] {short_summary or '(no summary)'}"

    body_lines = [
        "Hello Support Team,",
        "",
        "Please see the request details below:",
        "",
        f"Requester : {your_name or 'N/A'} <{your_email or 'N/A'}>",
        f"Category  : {category}",
        f"Country   : {country_name} ({iso})" if iso else f"Country   : {country_name}",
        "",
        "Summary",
        "-------",
        short_summary or "(no summary provided)",
        "",
        "Details",
        "-------",
        details or "(no details provided)",
        "",
        "Attachments / Links",
        "-------------------",
        "(Add any attachments after the compose window opens.)",
        "",
        "--",
        "Sent from TomTom Tax Agent Support page",
    ]
    body = "\n".join(body_lines)

    to_emails = [e.strip() for e in to_list.split(",") if e.strip()]
    cc_emails = [e.strip() for e in cc_list.split(",") if e.strip()]

    # Primary: Outlook Web compose deeplink + fallback mailto
    owa_url = build_outlook_web_compose_url(to_emails, cc_emails, subject, body)
    mailto_url = build_mailto_fallback(to_emails, cc_emails, subject, body)

    st.success("Please click on the button below.")
    st.markdown(
        f"""
        <script>
          try {{
            window.open("{owa_url}", "_blank");
          }} catch (e) {{
            console.warn("Popup blocked:", e);
          }}
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("Open in Outlook Web", owa_url, use_container_width=True)

# Optional: small help
with st.expander("Why Outlook Web?"):
    st.write(
        "Using Outlook on the web ensures a consistent experience. We pre-fill the **To**, **CC**, "
        "**Subject**, and **Body** so you can just review and send. If your browser blocks pop-ups, "
        "click the **Open in Outlook Web** button."
    )
