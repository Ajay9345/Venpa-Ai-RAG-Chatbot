import streamlit as st
from datetime import datetime
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from src.retriever import get_retriever
from src.rag import run_rag
from src.memory import ConversationMemory
from src.loader import load_documents

st.set_page_config(
    page_title="Venpa AI Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Shell ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: #0A0A0A !important;
    color: #FFFFFF;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Kill ALL Streamlit top spacing ── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
.main, .main > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

.main .block-container,
[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

/* Lock page — no outer scroll */
html, body {
    overflow: hidden !important;
    height: 100% !important;
}
[data-testid="stAppViewContainer"] {
    height: 100vh !important;
    overflow: hidden !important;
}
[data-testid="stMain"] {
    height: 100vh !important;
    overflow: hidden !important;
}

/* ─────────────────────────────────────────
   SIDEBAR
───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0D0D0D !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    width: 256px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 !important;
}

/* ── Sidebar inner ── */
.sidebar-inner {
    padding: 20px 16px 24px;
    display: flex;
    flex-direction: column;
    height: 100vh;
    gap: 0;
}

/* ── Logo block ── */
.sb-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 20px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 20px;
}
.sb-logomark {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    box-shadow: 0 0 0 1px rgba(79,70,229,0.4);
}
.sb-brand {
    font-size: 14px; font-weight: 600;
    color: #fff; letter-spacing: -0.02em;
}
.sb-brand-sub {
    font-size: 9.5px; font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.28); letter-spacing: 0.08em;
    margin-top: 1px; text-transform: uppercase;
}

/* ── New chat button ── */
.sb-newchat {
    display: flex; align-items: center; gap: 8px;
    background: rgba(79,70,229,0.12);
    border: 1px solid rgba(79,70,229,0.3);
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px; font-weight: 500;
    color: rgba(255,255,255,0.8);
    cursor: pointer;
    margin-bottom: 24px;
    transition: all 0.18s ease;
    width: 100%;
    text-align: left;
}
.sb-newchat:hover {
    background: rgba(79,70,229,0.2);
    border-color: rgba(79,70,229,0.5);
    color: #fff;
}
.sb-newchat-icon {
    width: 16px; height: 16px;
    background: rgba(79,70,229,0.5);
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; flex-shrink: 0;
}

/* ── Sidebar section label ── */
.sb-section-label {
    font-size: 9.5px; font-weight: 600;
    color: rgba(255,255,255,0.22);
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0 4px;
    margin-bottom: 6px;
}

/* ── Sidebar nav item ── */
.sb-nav-item {
    display: flex; align-items: center; gap: 9px;
    padding: 7px 10px;
    border-radius: 7px;
    font-size: 13px; font-weight: 450;
    color: rgba(255,255,255,0.45);
    cursor: pointer;
    transition: all 0.15s ease;
    margin-bottom: 1px;
}
.sb-nav-item:hover {
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.75);
}
.sb-nav-item.active {
    background: rgba(79,70,229,0.1);
    color: rgba(255,255,255,0.88);
    border: 1px solid rgba(79,70,229,0.2);
}
.sb-nav-icon {
    font-size: 13px; width: 16px; text-align: center; flex-shrink: 0;
}
.sb-nav-gap { margin-bottom: 16px; }

/* ── KB Stats ── */
.sb-stats {
    margin-top: auto;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.sb-stat-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 5px 4px;
    font-size: 11px;
}
.sb-stat-label { color: rgba(255,255,255,0.28); }
.sb-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px; color: #10B981; font-weight: 500;
}
.sb-live-badge {
    display: flex; align-items: center; gap: 5px;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.18);
    border-radius: 6px; padding: 5px 9px; margin-top: 10px;
    font-size: 10.5px; font-family: 'JetBrains Mono', monospace;
    color: #10B981; letter-spacing: 0.06em;
}
.sb-live-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #10B981;
    animation: livepulse 2s ease infinite;
}
@keyframes livepulse {
    0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}

/* ─────────────────────────────────────────
   TOP HEADER BAR
───────────────────────────────────────── */
.v-header {
    display: flex; align-items: center;
    padding: 14px 28px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(10,10,10,0.9);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
    gap: 16px;
}
.v-header-left {
    display: flex; align-items: center; gap: 10px;
    flex: 1;
}
.v-header-logomark {
    width: 26px; height: 26px;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px;
}
.v-header-brand {
    font-size: 13px; font-weight: 600; color: #fff;
    letter-spacing: -0.02em;
}
.v-header-divider {
    width: 1px; height: 18px;
    background: rgba(255,255,255,0.1);
    margin: 0 4px;
}
.v-header-title {
    font-size: 13px; font-weight: 500; color: rgba(255,255,255,0.6);
}

.v-header-center {
    text-align: center; flex: 1.4;
}
.v-header-center-title {
    font-size: 14px; font-weight: 600; color: #fff;
    letter-spacing: -0.02em;
}
.v-header-center-sub {
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.25); letter-spacing: 0.07em;
    text-transform: uppercase; margin-top: 1px;
}

.v-header-right {
    display: flex; align-items: center; gap: 8px;
    flex: 1; justify-content: flex-end;
}
.v-indicator {
    display: flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px; padding: 4px 9px;
    font-size: 10.5px; font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.35); letter-spacing: 0.04em;
}
.v-indicator.green { color: #10B981; border-color: rgba(16,185,129,0.2); background: rgba(16,185,129,0.06); }
.v-indicator.blue { color: #60A5FA; border-color: rgba(96,165,250,0.2); background: rgba(96,165,250,0.06); }
.v-ind-dot {
    width: 4px; height: 4px; border-radius: 50%;
    background: currentColor;
    animation: livepulse 2.5s ease infinite;
}

/* ─────────────────────────────────────────
   MAIN CONTENT AREA
───────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
    overflow: hidden !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
}

[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    overflow: hidden !important;
    padding: 0 !important;
}

/* Right panel scroll */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
    border-left: 1px solid rgba(255,255,255,0.06);
    overflow-y: auto !important;
    padding: 24px 18px !important;
}

/* Input bar stays at bottom of center col */
[data-testid="stBottom"] {
    padding: 0 36px 20px !important;
}

/* Style the native scroll container */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: none !important;
    background: transparent !important;
    padding: 0 36px !important;
}

/* Center column padding */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    padding: 24px 36px 0 !important;
}

/* ─────────────────────────────────────────
   HERO
───────────────────────────────────────── */
.v-hero {
    padding-bottom: 32px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 28px;
}
.v-hero-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(79,70,229,0.1);
    border: 1px solid rgba(79,70,229,0.25);
    border-radius: 999px; padding: 4px 12px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    color: #818CF8; letter-spacing: 0.06em;
    margin-bottom: 20px;
}
.v-hero-title {
    font-size: 2rem; font-weight: 700;
    letter-spacing: -0.04em; color: #fff;
    line-height: 1.15; margin-bottom: 12px;
}
.v-hero-title em {
    font-style: normal;
    background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.v-hero-sub {
    font-size: 0.88rem; color: rgba(255,255,255,0.38);
    line-height: 1.75; max-width: 560px;
}

/* ── Metrics ── */
.v-metrics {
    display: flex; gap: 0;
    margin: 24px 0 28px;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; overflow: hidden;
}
.v-metric {
    flex: 1; padding: 14px 18px;
    border-right: 1px solid rgba(255,255,255,0.07);
    position: relative;
}
.v-metric:last-child { border-right: none; }
.v-metric-val {
    font-size: 1.4rem; font-weight: 700;
    color: #fff; letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 4px;
}
.v-metric-val span {
    font-size: 0.85rem; font-weight: 500;
    color: #818CF8;
}
.v-metric-label {
    font-size: 10.5px; color: rgba(255,255,255,0.28);
    letter-spacing: 0.03em;
}

/* ── Quick Action Cards ── */
.v-cards {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
}
.v-card {
    background: #111111;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative; overflow: hidden;
}
.v-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,70,229,0.4), transparent);
    opacity: 0;
    transition: opacity 0.2s ease;
}
.v-card:hover {
    border-color: rgba(79,70,229,0.3);
    background: #141414;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.v-card:hover::before { opacity: 1; }
.v-card-icon {
    font-size: 18px; margin-bottom: 10px;
    width: 34px; height: 34px;
    background: rgba(79,70,229,0.1);
    border: 1px solid rgba(79,70,229,0.2);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
}
.v-card-title {
    font-size: 12.5px; font-weight: 600; color: rgba(255,255,255,0.8);
    margin-bottom: 4px;
}
.v-card-desc {
    font-size: 11px; color: rgba(255,255,255,0.28);
    line-height: 1.5;
}
.v-card-arrow {
    position: absolute; top: 12px; right: 12px;
    font-size: 12px; color: rgba(255,255,255,0.18);
    transition: all 0.2s ease;
}
.v-card:hover .v-card-arrow {
    color: rgba(79,70,229,0.7);
    transform: translate(2px,-2px);
}

/* ─────────────────────────────────────────
   CHAT AREA
───────────────────────────────────────── */
.v-chat-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
}
.v-chat-title {
    font-size: 12px; font-weight: 600;
    color: rgba(255,255,255,0.3);
    letter-spacing: 0.06em; text-transform: uppercase;
}
.v-msg-count {
    font-size: 10.5px; font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 5px; padding: 2px 7px;
}

/* Back button */
[data-testid="stButton"] button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.5) !important;
    font-size: 12px !important;
    padding: 5px 12px !important;
    margin-bottom: 12px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stButton"] button:hover {
    background: rgba(79,70,229,0.1) !important;
    border-color: rgba(79,70,229,0.3) !important;
    color: rgba(255,255,255,0.8) !important;
}

/* Streamlit chat message overrides */
.stChatMessage {
    background: transparent !important;
    padding: 6px 0 !important;
    border: none !important;
    box-shadow: none !important;
    animation: msgfadein 0.25s ease;
}
@keyframes msgfadein {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Assistant bubble */
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"])
[data-testid="stChatMessageContent"] {
    background: #111111 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: 1px solid rgba(79,70,229,0.18) !important;
    border-radius: 4px 12px 12px 12px !important;
    color: rgba(255,255,255,0.82) !important;
    font-size: 0.862rem !important;
    padding: 14px 16px !important;
    max-width: 82% !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
    line-height: 1.75 !important;
    position: relative;
}

/* User bubble */
.stChatMessage:has([data-testid="chatAvatarIcon-user"])
[data-testid="stChatMessageContent"] {
    background: rgba(79,70,229,0.12) !important;
    border: 1px solid rgba(79,70,229,0.22) !important;
    border-radius: 12px 4px 12px 12px !important;
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.862rem !important;
    padding: 12px 16px !important;
    max-width: 76% !important;
    margin-left: auto !important;
    box-shadow: none !important;
    line-height: 1.75 !important;
}

/* Avatars */
[data-testid="chatAvatarIcon-assistant"] {
    background: #1A1745 !important;
    border: 1px solid rgba(79,70,229,0.3) !important;
    border-radius: 8px !important;
    color: #818CF8 !important;
    font-size: 13px !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: #1C1C1C !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.4) !important;
}

/* ─────────────────────────────────────────
   CHAT INPUT — pill style with glow
───────────────────────────────────────── */
[data-testid="stBottom"] {
    background: linear-gradient(to top, #0A0A0A 70%, transparent) !important;
    padding: 0 36px 20px !important;
    flex-shrink: 0 !important;
    z-index: 10 !important;
}
/* nuke every layer inside the chat input */
[data-testid="stChatInput"],
[data-testid="stChatInput"] *:not(textarea):not(button):not(button *) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 !important;
}

/* pill wrapper — the one visible container */
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 999px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(16px) !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    padding: 4px 8px 4px 20px !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(79,70,229,0.6) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15), 0 2px 24px rgba(79,70,229,0.18) !important;
}

/* textarea */
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: rgba(255,255,255,0.88) !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    line-height: 1.6 !important;
    padding: 10px 0 !important;
    caret-color: #818CF8 !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.22) !important;
    font-style: italic !important;
}

/* send button */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    border-radius: 999px !important;
    border: none !important;
    min-width: 36px !important;
    height: 36px !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.4) !important;
    transition: all 0.18s ease !important;
    flex-shrink: 0 !important;
    padding: 0 !important;
}
[data-testid="stChatInput"] button:hover {
    background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%) !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.55) !important;
    transform: scale(1.05) !important;
}

/* ─────────────────────────────────────────
   RIGHT PANEL
───────────────────────────────────────── */
.rp-section-label {
    font-size: 9.5px; font-weight: 600;
    color: rgba(255,255,255,0.22);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 10px;
}
.rp-chip {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 9px 10px;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 5px;
    transition: all 0.15s ease;
    background: #111;
}
.rp-chip:hover {
    border-color: rgba(79,70,229,0.3);
    background: rgba(79,70,229,0.06);
}
.rp-chip-icon {
    font-size: 11px; opacity: 0.4; flex-shrink: 0; margin-top: 1px;
}
.rp-chip-text {
    font-size: 11.5px; color: rgba(255,255,255,0.48);
    line-height: 1.5;
}
.rp-chip:hover .rp-chip-text { color: rgba(255,255,255,0.72); }

.rp-service-card {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 10px;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    margin-bottom: 5px;
    background: #111;
}
.rp-service-icon {
    width: 28px; height: 28px; flex-shrink: 0;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
}
.rp-service-name {
    font-size: 12px; font-weight: 500;
    color: rgba(255,255,255,0.6);
}
.rp-service-badge {
    margin-left: auto; flex-shrink: 0;
    font-size: 9.5px; font-family: 'JetBrains Mono', monospace;
    color: #10B981;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.15);
    border-radius: 4px; padding: 2px 6px;
}

.rp-divider {
    border: none; border-top: 1px solid rgba(255,255,255,0.05);
    margin: 18px 0;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4F46E5 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb {
    background: rgba(79,70,229,0.2); border-radius: 99px;
}
::-webkit-scrollbar-track { background: transparent; }
</style>
""", unsafe_allow_html=True)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_rag():
    docs = load_documents()
    embedding_model = get_embedding_model()
    vectordb = load_vector_store(embedding_model)
    retriever = get_retriever(vectordb, docs)
    return retriever

with st.spinner("Connecting knowledge base…"):
    retriever = load_rag()

with st.sidebar:
    st.markdown("""
    <div class="sidebar-inner">
      <!-- Logo -->
      <div class="sb-logo">
        <div class="sb-logomark">⚡</div>
        <div>
          <div class="sb-brand">Venpa AI</div>
          <div class="sb-brand-sub">Autonomous AI Agency</div>
        </div>
      </div>

      <!-- New Chat -->
      <div class="sb-newchat">
        <div class="sb-newchat-icon">✦</div>
        New conversation
      </div>

      <!-- Services Nav -->
      <div class="sb-section-label">Services</div>
      <div class="sb-nav-item active">
        <span class="sb-nav-icon">🤖</span> AI Agents
      </div>
      <div class="sb-nav-item">
        <span class="sb-nav-icon">🎙️</span> Voice Agents
      </div>
      <div class="sb-nav-item">
        <span class="sb-nav-icon">💬</span> Chatbots
      </div>
      <div class="sb-nav-item">
        <span class="sb-nav-icon">🔍</span> GEO / AEO
      </div>
      <div class="sb-nav-item">
        <span class="sb-nav-icon">🗄️</span> RAG Systems
      </div>
      <div class="sb-nav-item">
        <span class="sb-nav-icon">⚙️</span> AI Automation
      </div>
      <div class="sb-nav-item sb-nav-gap">
        <span class="sb-nav-icon">👤</span> AI Twin
      </div>

      <!-- Company Nav -->
      <div class="sb-section-label">Company</div>
      <div class="sb-nav-item">
        <span class="sb-nav-icon">🏢</span> About Venpa AI
      </div>
      <div class="sb-nav-item">
        <span class="sb-nav-icon">👨‍💼</span> Founder
      </div>
      <div class="sb-nav-item sb-nav-gap">
        <span class="sb-nav-icon">📩</span> Contact
      </div>

      <!-- KB Stats -->
      <div class="sb-stats">
        <div class="sb-section-label">Knowledge Base</div>
        <div class="sb-stat-row">
          <span class="sb-stat-label">Documents</span>
          <span class="sb-stat-val">Connected</span>
        </div>
        <div class="sb-stat-row">
          <span class="sb-stat-label">Retrieval</span>
          <span class="sb-stat-val">Semantic</span>
        </div>
        <div class="sb-stat-row">
          <span class="sb-stat-label">Memory</span>
          <span class="sb-stat-val">5 turns</span>
        </div>
        <div class="sb-live-badge">
          <div class="sb-live-dot"></div>
          SYSTEM ONLINE
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="v-header">
  <div class="v-header-left">
    <div class="v-header-logomark">⚡</div>
    <div class="v-header-brand">Venpa AI</div>
    <div class="v-header-divider"></div>
    <div class="v-header-title">Copilot</div>
  </div>
  <div class="v-header-center">
    <div class="v-header-center-title">Venpa AI Copilot</div>
    <div class="v-header-center-sub">Autonomous Operating Layer</div>
  </div>
  <div class="v-header-right">
    <div class="v-indicator green">
      <div class="v-ind-dot"></div> Live
    </div>
    <div class="v-indicator blue">
      <div class="v-ind-dot"></div> KB Connected
    </div>
    <div class="v-indicator">
      &lt;2s Avg Response
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

col_center, col_right = st.columns([3.2, 1], gap="small")

with col_right:
    st.markdown('<div style="padding: 24px 0 0;">', unsafe_allow_html=True)

    st.markdown('<div class="rp-section-label">Suggested Questions</div>', unsafe_allow_html=True)
    questions = [
        ("💼", "What services does Venpa AI offer?"),
        ("🔍", "Explain GEO and AEO."),
        ("🎙️", "How does an AI Voice Agent work?"),
        ("👤", "What is an AI Twin?"),
        ("⚙️", "How can AI automation help my business?"),
        ("📊", "What ROI can I expect from AI?"),
    ]
    for icon, q in questions:
        st.markdown(f"""
        <div class="rp-chip">
          <span class="rp-chip-icon">{icon}</span>
          <span class="rp-chip-text">{q}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="rp-divider">', unsafe_allow_html=True)

    st.markdown('<div class="rp-section-label">Popular Services</div>', unsafe_allow_html=True)
    services = [
        ("🤖", "#1A1745", "AI Agents", "Core"),
        ("🔍", "#14261A", "GEO / AEO", "Hot"),
        ("🎙️", "#261A14", "Voice Agents", "New"),
        ("🗄️", "#1A2026", "RAG Systems", "Core"),
    ]
    for icon, bg, name, badge in services:
        st.markdown(f"""
        <div class="rp-service-card">
          <div class="rp-service-icon" style="background:{bg};">{icon}</div>
          <div class="rp-service-name">{name}</div>
          <div class="rp-service-badge">{badge}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_center:
    chat_height = 560 if len(st.session_state.messages) == 0 else 560
    chat_container = st.container(height=chat_height, border=False)

    with chat_container:
     
     if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="v-hero">
          <div class="v-hero-eyebrow">
            ⚡ Autonomous AI Copilot
          </div>
          <div class="v-hero-title">
            Your <em>Autonomous AI</em><br>Consultant
          </div>
          <div class="v-hero-sub">
            Explore AI agents, automation systems, growth infrastructure,
            RAG deployments, and enterprise AI transformation — all in one conversation.
          </div>

          <div class="v-metrics">
            <div class="v-metric">
              <div class="v-metric-val">97<span>%</span></div>
              <div class="v-metric-label">Client Retention</div>
            </div>
            <div class="v-metric">
              <div class="v-metric-val">14<span>d</span></div>
              <div class="v-metric-label">Prototype Delivery</div>
            </div>
            <div class="v-metric">
              <div class="v-metric-val">99.9<span>%</span></div>
              <div class="v-metric-label">Uptime SLA</div>
            </div>
            <div class="v-metric">
              <div class="v-metric-val">3<span>×</span></div>
              <div class="v-metric-label">Average ROI</div>
            </div>
          </div>

          <div class="v-cards">
            <div class="v-card">
              <div class="v-card-arrow">↗</div>
              <div class="v-card-icon">🔭</div>
              <div class="v-card-title">Explore Services</div>
              <div class="v-card-desc">Full stack AI offerings for enterprise teams</div>
            </div>
            <div class="v-card">
              <div class="v-card-arrow">↗</div>
              <div class="v-card-icon">🧪</div>
              <div class="v-card-title">AI Readiness Audit</div>
              <div class="v-card-desc">Assess your org's current AI maturity</div>
            </div>
            <div class="v-card">
              <div class="v-card-arrow">↗</div>
              <div class="v-card-icon">⚡</div>
              <div class="v-card-title">Automation Wins</div>
              <div class="v-card-desc">Identify high-impact automation opportunities</div>
            </div>
            <div class="v-card">
              <div class="v-card-arrow">↗</div>
              <div class="v-card-icon">📅</div>
              <div class="v-card-title">Strategy Call</div>
              <div class="v-card-desc">Book a session with the Venpa AI team</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

     if len(st.session_state.messages) > 0:
         msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
         st.markdown(f"""
         <div class="v-chat-header">
           <span class="v-chat-title">Conversation</span>
           <span class="v-msg-count">{msg_count} message{'s' if msg_count != 1 else ''}</span>
         </div>
         """, unsafe_allow_html=True)
         if st.button("← Back to Home", key="back_home"):
             st.session_state.messages = []
             st.session_state.memory = ConversationMemory(max_turns=5)
             st.rerun()

     if len(st.session_state.messages) == 0:
         with st.chat_message("assistant"):
             st.markdown(
                 "Hello — I'm Venpa's AI Copilot. I can walk you through our autonomous "
                 "AI agents, voice agents, GEO & AEO strategies, RAG deployments, and "
                 "how we design enterprise AI transformation. What would you like to explore?"
             )

     for msg in st.session_state.messages:
         with st.chat_message(msg["role"]):
             st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about Venpa AI…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner(""):
                answer = run_rag(user_input, retriever, st.session_state.memory)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()