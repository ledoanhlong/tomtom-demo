from __future__ import annotations
import pandas as pd
import streamlit as st

APP_TITLE = "Country Overview"
st.set_page_config(page_title=APP_TITLE, page_icon="🌍", initial_sidebar_state="expanded")

# --- (Optional) light CSS to match main app ---
def inject_light_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --secondary-background-color: #f5f7fb;
                --border-color: #e5e7eb;
            }
            section[data-testid="stSidebar"] {
                background: var(--secondary-background-color) !important;
                border-right: 1px solid var(--border-color) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_light_css()

# --- Helpers ---
def iso_to_flag(iso2: str) -> str:
    iso = (iso2 or "").upper()
    if len(iso) != 2 or not iso.isalpha():
        return "🏳️"
    return chr(ord(iso[0]) - 65 + 0x1F1E6) + chr(ord(iso[1]) - 65 + 0x1F1E6)

ISO_TO_NAME = {
    "NL": "Netherlands", "BE": "Belgium", "DE": "Germany", "FR": "France",
    "GB": "United Kingdom", "ES": "Spain", "IT": "Italy", "PL": "Poland",
    "US": "United States", "CA": "Canada",
}

# --- Read selection (query param first, then session) ---
country_iso = st.query_params.get("country") or st.session_state.get("selected_country_iso")
country_name = st.session_state.get("selected_country_name") or (ISO_TO_NAME.get((country_iso or "").upper()) if country_iso else None)

if not country_iso or not country_name:
    st.warning("No country selected yet. Go back and choose a country from the sidebar.")
    if st.button("⬅️ Back to Chat", use_container_width=True):
        try:
            st.query_params.clear()
            st.switch_page("Home.py")
        except Exception:
            st.rerun()
    st.stop()

flag = iso_to_flag(country_iso)

# --- Sidebar: Back button ---
with st.sidebar:
    if st.button("⬅️ Back to Chat", use_container_width=True):
        try:
            st.query_params.clear()
            st.switch_page("Home.py")
        except Exception:
            st.rerun()

# --- Header ---
st.title(f"{flag} {country_name}")
st.caption(f"ISO code: {country_iso.upper()}")

# --- Demo data: deadlines ---
st.markdown("---")
st.subheader("📅 Key Deadlines (Demo)")
deadlines = pd.DataFrame(
    [
        {"Deadline": "2025-09-15", "Type": "VAT Return", "Frequency": "Monthly", "Notes": "Standard filing"},
        {"Deadline": "2025-10-31", "Type": "CIT Estimate", "Frequency": "Quarterly", "Notes": "Q3 estimate due"},
        {"Deadline": "2025-12-31", "Type": "Intrastat Report", "Frequency": "Monthly", "Notes": "Goods arrivals"},
    ]
)
st.dataframe(deadlines, use_container_width=True, hide_index=True)

# --- Demo data: obligations ---
st.markdown("---")
st.subheader("🧾 Tax Obligations (Demo)")
obligations = pd.DataFrame(
    [
        {"Obligation": "Corporate Income Tax", "Status": "Active", "Threshold/Rate": "Base rate 25% (demo)"},
        {"Obligation": "Value Added Tax (VAT)", "Status": "Active", "Threshold/Rate": "Standard 21% (demo)"},
        {"Obligation": "WHT on Dividends", "Status": "Applies", "Threshold/Rate": "15% treaty-dependent (demo)"},
    ]
)
st.dataframe(obligations, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("ℹ️ Notes")
st.markdown(
    """
    *Demo only.* Replace with live data sources.  
    This page reads the country from the URL `?country=XX` (via `st.query_params`)
    and also uses session state for the chosen country name.
    """
)
