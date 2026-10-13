from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from modules.agreement_engine import top_k_agreement
from modules.database import clear_history, initialize_database, read_experiments, save_experiment
from modules.dataset_manager import generate_demo_data, load_best_available_dataset, load_csv, map_binary_labels, normalize_columns
from modules.lime_engine import LimeEngine
from modules.model_engine import load_artifacts, save_training_result, train_intrusion_model
from modules.permutation_engine import calculate_permutation_importance
from modules.prediction_engine import predict_transformed
from modules.report_generator import generate_html_report
from modules.shap_engine import ShapEngine
from modules.visualization import contribution_chart, confusion_matrix_chart, importance_chart, radar_chart
from modules.xai_evaluator import evaluate_explainer

st.set_page_config(
    page_title="E-XAI | Explainable Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

pio.templates["exai_premium"] = go.layout.Template(
    layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(4,14,28,0.34)",
        "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#cbd8e6", "size": 13},
        "title": {"font": {"color": "#f5f9ff", "size": 19}, "x": 0.02, "xanchor": "left"},
        "colorway": ["#5eead4", "#38bdf8", "#a78bfa", "#fb7185", "#fbbf24"],
        "xaxis": {"gridcolor": "rgba(148,163,184,.10)", "zerolinecolor": "rgba(148,163,184,.18)"},
        "yaxis": {"gridcolor": "rgba(148,163,184,.10)", "zerolinecolor": "rgba(148,163,184,.18)"},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#b8c7d9"}},
        "hoverlabel": {"bgcolor": "#0b1728", "bordercolor": "#24415f", "font": {"color": "#f8fbff"}},
        "margin": {"l": 40, "r": 24, "t": 64, "b": 42},
    }
)
pio.templates.default = "exai_premium"

st.markdown("""
<style>
:root{
  --navy-950:#030914;--navy-900:#07111f;--panel:rgba(10,25,43,.74);
  --panel-strong:rgba(12,31,52,.94);--line:rgba(125,211,252,.16);
  --text:#eef6ff;--muted:#8fa4ba;--cyan:#38bdf8;--teal:#5eead4;
  --violet:#a78bfa;--danger:#fb7185;--warning:#fbbf24;
}
html,body,[class*="css"]{font-family:Inter,"Segoe UI",system-ui,sans-serif}
[data-testid="stAppViewContainer"]{
  color:var(--text);
  background:
    radial-gradient(circle at 78% -15%,rgba(56,189,248,.16),transparent 34%),
    radial-gradient(circle at 8% 32%,rgba(94,234,212,.08),transparent 28%),
    linear-gradient(150deg,#030914 0%,#07111f 48%,#06101c 100%);
}
[data-testid="stAppViewContainer"]:before{
  content:"";position:fixed;inset:0;pointer-events:none;opacity:.18;
  background-image:linear-gradient(rgba(125,211,252,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(125,211,252,.055) 1px,transparent 1px);
  background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 78%);
}
[data-testid="stHeader"]{height:0;background:transparent;border:0}
[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stMainBlockContainer"]{max-width:1500px;padding-top:1.25rem;padding-bottom:5rem}
[data-testid="stSidebar"]{width:286px!important;min-width:286px!important;background:linear-gradient(180deg,rgba(5,15,28,.98),rgba(7,20,35,.98));border-right:1px solid var(--line)}
[data-testid="stSidebarHeader"]{height:2.6rem;min-height:2.6rem;padding:.25rem .65rem}
[data-testid="stSidebarContent"]{padding:.3rem .85rem 1rem}
.brand-card{padding:1.1rem 1rem 1rem;margin:.15rem 0 1.1rem;border-radius:18px;background:linear-gradient(145deg,rgba(18,52,80,.82),rgba(9,27,46,.76));border:1px solid rgba(94,234,212,.22);box-shadow:0 18px 48px rgba(0,0,0,.22)}
.brand-row{display:flex;align-items:center;gap:.75rem}.brand-mark{width:42px;height:42px;display:grid;place-items:center;border-radius:13px;background:linear-gradient(145deg,#0ea5e9,#14b8a6);box-shadow:0 0 26px rgba(56,189,248,.25);font-size:1.25rem}
.brand-name{font-weight:800;letter-spacing:.05em;font-size:1.05rem}.brand-copy{color:var(--muted);font-size:.72rem;line-height:1.45;margin-top:.65rem}.nav-label{color:#66839e;font-size:.64rem;font-weight:800;letter-spacing:.18em;margin:.25rem .55rem .5rem}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.22rem}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.62rem .7rem;border-radius:10px;border:1px solid transparent;transition:.2s ease}
[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none!important}
[data-testid="stSidebar"] [role="radiogroup"] label>div:last-child{width:100%}
[data-testid="stSidebar"] [role="radiogroup"] label p{white-space:nowrap;font-size:.88rem}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(56,189,248,.07);border-color:rgba(56,189,248,.12);transform:translateX(2px)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,rgba(14,165,233,.20),rgba(20,184,166,.07));border-color:rgba(56,189,248,.28);box-shadow:inset 3px 0 0 #38bdf8}
.model-card{margin-top:1.1rem;padding:.85rem;border-radius:14px;background:rgba(6,20,35,.74);border:1px solid rgba(94,234,212,.15)}
.model-card .online{display:flex;align-items:center;gap:.45rem;color:#a7f3d0;font-size:.72rem;font-weight:800;letter-spacing:.08em}.status-dot{width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 12px #34d399}.model-name{font-weight:700;color:#e7f5ff;margin:.5rem 0 .15rem}.model-meta{font-size:.69rem;color:#71889f}
.hero{position:relative;overflow:hidden;padding:1.55rem 1.7rem;border:1px solid rgba(125,211,252,.18);border-radius:22px;background:linear-gradient(125deg,rgba(13,38,64,.92),rgba(8,24,42,.72));box-shadow:0 22px 70px rgba(0,0,0,.22);margin-bottom:1rem}
.hero:after{content:"";position:absolute;width:280px;height:280px;border-radius:50%;right:-90px;top:-170px;background:radial-gradient(circle,rgba(56,189,248,.24),transparent 66%)}
.eyebrow{font-size:.66rem;font-weight:800;letter-spacing:.18em;color:var(--teal);text-transform:uppercase;margin-bottom:.45rem}.hero h1{position:relative;margin:0;color:#f6fbff;font-size:clamp(1.8rem,3vw,2.6rem);letter-spacing:-.035em;line-height:1.06}.hero-copy{position:relative;color:#9eb2c6;margin-top:.55rem;font-size:.9rem}.hero-meta{display:flex;gap:.45rem;flex-wrap:wrap;margin-top:1rem}.hero-chip{font-size:.66rem;font-weight:700;letter-spacing:.05em;color:#b9d5e8;border:1px solid rgba(125,211,252,.16);background:rgba(2,10,20,.28);padding:.34rem .58rem;border-radius:999px}.hero-chip.live{color:#a7f3d0;border-color:rgba(52,211,153,.25)}
.notice{display:flex;align-items:center;gap:.7rem;padding:.72rem .9rem;border:1px solid rgba(251,191,36,.22);background:rgba(78,57,13,.34);border-radius:12px;color:#f8df9c;margin:.8rem 0 1.15rem;font-size:.82rem}.notice-icon{width:25px;height:25px;display:grid;place-items:center;border-radius:8px;background:rgba(251,191,36,.13)}
.section-head{margin:1.45rem 0 .78rem}.section-tag{color:var(--cyan);font-size:.64rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}.section-title{font-size:1.12rem;font-weight:750;color:#edf7ff;margin-top:.18rem}.section-copy{font-size:.79rem;color:var(--muted);margin-top:.16rem}
[data-testid="stMetric"]{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(13,36,59,.86),rgba(8,25,43,.76));border:1px solid var(--line);border-radius:16px;padding:1rem 1.05rem;box-shadow:0 12px 30px rgba(0,0,0,.14);min-height:106px}
[data-testid="stMetric"]:before{content:"";position:absolute;left:0;top:0;right:0;height:2px;background:linear-gradient(90deg,var(--cyan),var(--teal),transparent)}
[data-testid="stMetricLabel"]{color:#8da5bb;font-size:.7rem;font-weight:750;letter-spacing:.06em;text-transform:uppercase}
[data-testid="stMetricValue"]{color:#f5f9ff;font-weight:780;letter-spacing:-.03em}
.kpi-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.78rem;margin:.25rem 0 .5rem}.kpi-card{position:relative;overflow:hidden;min-height:105px;padding:1rem 1.05rem;border-radius:16px;background:linear-gradient(145deg,rgba(13,36,59,.86),rgba(8,25,43,.76));border:1px solid var(--line);box-shadow:0 12px 30px rgba(0,0,0,.14)}.kpi-card:before{content:"";position:absolute;left:0;top:0;right:0;height:2px;background:linear-gradient(90deg,var(--cyan),var(--teal),transparent)}.kpi-label{color:#8da5bb;font-size:.7rem;font-weight:750;letter-spacing:.06em;text-transform:uppercase}.kpi-value{color:#f5f9ff;font-size:1.85rem;line-height:1.2;font-weight:800;letter-spacing:-.035em;margin-top:.65rem;overflow-wrap:anywhere}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stJson"]{background:rgba(8,22,38,.54);border:1px solid rgba(125,211,252,.11);border-radius:16px;overflow:hidden;box-shadow:0 14px 36px rgba(0,0,0,.11)}
[data-testid="stForm"]{background:linear-gradient(145deg,rgba(11,31,52,.76),rgba(7,21,37,.68));border:1px solid var(--line);border-radius:18px;padding:1.15rem;box-shadow:0 16px 45px rgba(0,0,0,.16)}
.stButton>button,[data-testid="stDownloadButton"] button,[data-testid="stFormSubmitButton"] button{min-height:2.65rem;border-radius:11px;border:1px solid rgba(56,189,248,.35);background:linear-gradient(135deg,#075985,#0f766e);color:white;font-weight:750;letter-spacing:.025em;box-shadow:0 10px 25px rgba(14,116,144,.19);transition:.18s ease}
.stButton>button:hover,[data-testid="stDownloadButton"] button:hover,[data-testid="stFormSubmitButton"] button:hover{border-color:#67e8f9;box-shadow:0 12px 34px rgba(14,165,233,.26);transform:translateY(-1px);color:white}
.stButton>button:disabled{opacity:.42;box-shadow:none}
[data-baseweb="tab-list"]{gap:.35rem;background:rgba(5,16,29,.45);padding:.35rem;border-radius:12px;border:1px solid rgba(125,211,252,.1)}
[data-baseweb="tab"]{height:2.6rem;border-radius:9px;padding:0 1rem}[aria-selected="true"][data-baseweb="tab"]{background:rgba(56,189,248,.13);color:#e9f8ff}
[data-baseweb="input"]>div,[data-baseweb="select"]>div,[data-baseweb="base-input"]{background:rgba(4,15,28,.74)!important;border-color:rgba(125,211,252,.16)!important;border-radius:10px!important}
[data-testid="stFileUploaderDropzone"]{background:rgba(7,24,41,.72);border:1px dashed rgba(94,234,212,.32);border-radius:14px}
[data-testid="stExpander"]{background:rgba(8,24,41,.5);border:1px solid rgba(125,211,252,.12);border-radius:14px!important;overflow:hidden}
[data-testid="stAlert"]{border-radius:12px;border-width:1px;background:rgba(10,27,46,.72)}
.pipeline{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:.5rem;margin:.45rem 0 1rem}.pipe-step{position:relative;padding:.82rem .65rem;border-radius:13px;background:rgba(10,30,50,.72);border:1px solid rgba(125,211,252,.12);min-height:92px}.pipe-step:not(:last-child):after{content:"›";position:absolute;right:-.47rem;top:34%;z-index:2;color:#54d8eb;font-size:1.25rem}.pipe-index{font-size:.6rem;color:#4e7894;font-weight:800}.pipe-icon{font-size:1.05rem;margin:.32rem 0}.pipe-name{font-size:.69rem;line-height:1.25;color:#d9e9f5;font-weight:700}
.prediction{padding:1.2rem 1.25rem;border-radius:18px;background:linear-gradient(135deg,rgba(13,41,65,.93),rgba(9,28,47,.85));border:1px solid rgba(56,189,248,.23);box-shadow:0 18px 48px rgba(0,0,0,.18)}
.prediction-top{display:flex;justify-content:space-between;align-items:center;gap:1rem}.prediction-label{font-size:1.55rem;font-weight:820;letter-spacing:-.025em}.risk-pill{font-size:.68rem;font-weight:850;letter-spacing:.1em;padding:.4rem .65rem;border-radius:999px;background:rgba(251,113,133,.11);border:1px solid rgba(251,113,133,.24);color:#fda4af}.prob-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}.prob-label{display:flex;justify-content:space-between;color:#9cb0c4;font-size:.73rem;margin-bottom:.36rem}.prob-track{height:7px;background:rgba(148,163,184,.12);border-radius:99px;overflow:hidden}.prob-fill{height:100%;border-radius:99px}.prob-fill.attack{background:linear-gradient(90deg,#f43f5e,#fb7185)}.prob-fill.benign{background:linear-gradient(90deg,#14b8a6,#5eead4)}
.score-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;margin:.7rem 0 1rem}.score-card{padding:1rem;border-radius:16px;background:rgba(8,27,45,.72);border:1px solid rgba(125,211,252,.13)}.score-head{display:flex;align-items:center;justify-content:space-between}.score-name{font-weight:800;color:#edf7ff}.score-quality{font-size:1.4rem;font-weight:820;color:var(--teal)}.score-meta{color:#829ab0;font-size:.7rem;margin-top:.3rem}.score-bar{height:5px;background:rgba(148,163,184,.12);border-radius:99px;overflow:hidden;margin-top:.75rem}.score-bar span{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--teal));border-radius:99px}
.method-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:.65rem 0 1.2rem}.method-card{padding:1rem;border-radius:15px;background:linear-gradient(145deg,rgba(12,34,55,.76),rgba(7,23,39,.68));border:1px solid rgba(125,211,252,.12);min-height:130px}.method-icon{width:31px;height:31px;display:grid;place-items:center;border-radius:9px;background:rgba(56,189,248,.11);color:#7dd3fc;margin-bottom:.65rem}.method-title{font-size:.86rem;font-weight:800;color:#edf7ff}.method-copy{font-size:.73rem;color:#8fa5ba;line-height:1.55;margin-top:.34rem}
h1,h2,h3{color:#eaf5ff;letter-spacing:-.02em}p{color:#aebfd0}hr{border-color:rgba(125,211,252,.1)!important}
::-webkit-scrollbar{width:9px;height:9px}::-webkit-scrollbar-track{background:#06101d}::-webkit-scrollbar-thumb{background:#183b56;border-radius:10px}::-webkit-scrollbar-thumb:hover{background:#245676}
@media(max-width:1100px){.pipeline{grid-template-columns:repeat(4,1fr)}.pipe-step:after{display:none}.method-grid{grid-template-columns:repeat(2,1fr)}.kpi-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:720px){[data-testid="stSidebar"]{width:280px!important;min-width:280px!important}[data-testid="stMainBlockContainer"]{padding-left:2rem!important;padding-right:1rem!important}.hero{padding:1.25rem}.pipeline{grid-template-columns:repeat(2,1fr)}.prob-row,.score-grid,.method-grid,.kpi-grid{grid-template-columns:1fr}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def resources():
    artifacts = load_artifacts()
    train = artifacts["train_reference"]["X"]
    artifacts["shap_engine"] = ShapEngine(artifacts["model"], train, artifacts["feature_names"])
    artifacts["lime_engine"] = LimeEngine(artifacts["model"], train, artifacts["feature_names"])
    return artifacts


@st.cache_data
def application_data(demo_mode: bool):
    if demo_mode:
        frame = normalize_columns(generate_demo_data())
        info = {"mode": "demo", "name": "Deterministic network-flow demonstration data", "files": ["generated in memory"]}
        return frame, info
    return load_best_available_dataset(prefer_real=True)


def get_context():
    try:
        artifacts = resources()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.code(".venv\\Scripts\\python.exe train_model.py")
        st.stop()
    metadata = artifacts["metadata"]
    frame, info = application_data(bool(metadata.get("demo_mode")))
    if "active_frame" not in st.session_state:
        st.session_state.active_frame = frame
        st.session_state.dataset_info = info
    return artifacts, st.session_state.active_frame, st.session_state.dataset_info


def ensure_selected(artifacts, frame):
    if "selected_row" not in st.session_state:
        transformed = artifacts["preprocessor"].transform(frame.iloc[[0]])[0]
        st.session_state.selected_row = transformed
        st.session_state.selected_raw = frame.iloc[[0]]
        st.session_state.sample_id = "dataset-0"
        st.session_state.prediction = predict_transformed(artifacts["model"], transformed)
    return st.session_state.selected_row, st.session_state.prediction


def kpis(items: list[tuple[str, str]], columns_per_row: int = 3):
    """Render a responsive KPI grid without uneven partial Streamlit rows."""
    del columns_per_row  # Grid density is controlled responsively in CSS.
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def section_header(title: str, copy: str, tag: str = "INTELLIGENCE"):
    st.markdown(
        f'<div class="section-head"><div class="section-tag">{tag}</div>'
        f'<div class="section-title">{title}</div><div class="section-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def pipeline_panel():
    stages = [
        ("01", "◉", "Network traffic"), ("02", "⌁", "Preprocessing"),
        ("03", "◆", "Black-box model"), ("04", "⌖", "Prediction"),
        ("05", "✦", "SHAP & LIME"), ("06", "◎", "E-XAI evaluation"),
        ("07", "▥", "Comparison"),
    ]
    cards = "".join(
        f'<div class="pipe-step"><div class="pipe-index">{index}</div><div class="pipe-icon">{icon}</div>'
        f'<div class="pipe-name">{name}</div></div>' for index, icon, name in stages
    )
    st.markdown(f'<div class="pipeline">{cards}</div>', unsafe_allow_html=True)


def prediction_panel(prediction: dict):
    malicious = bool(prediction["prediction"])
    label_color = "#fb7185" if malicious else "#5eead4"
    attack = prediction["attack_probability"] * 100
    benign = prediction["benign_probability"] * 100
    st.markdown(
        f'''<div class="prediction">
        <div class="prediction-top"><div><div class="section-tag">MODEL VERDICT</div>
        <div class="prediction-label" style="color:{label_color}">{prediction["label"]}</div></div>
        <div class="risk-pill">{prediction["risk"]} RISK</div></div>
        <div class="prob-row">
          <div><div class="prob-label"><span>Attack probability</span><b>{attack:.2f}%</b></div>
          <div class="prob-track"><div class="prob-fill attack" style="width:{attack:.2f}%"></div></div></div>
          <div><div class="prob-label"><span>Benign probability</span><b>{benign:.2f}%</b></div>
          <div class="prob-track"><div class="prob-fill benign" style="width:{benign:.2f}%"></div></div></div>
        </div></div>''',
        unsafe_allow_html=True,
    )


def evaluation_cards(results: list[dict]):
    cards = ""
    for result in results:
        quality = float(result["quality"])
        cards += (
            f'<div class="score-card"><div class="score-head"><div class="score-name">{result["explainer"]}</div>'
            f'<div class="score-quality">{quality:.3f}</div></div>'
            f'<div class="score-meta">Fidelity {result["fidelity"]:.3f} · Stability {result["stability"]:.3f} · '
            f'Runtime {result["runtime"]:.3f}s</div><div class="score-bar"><span style="width:{quality * 100:.1f}%"></span></div></div>'
        )
    st.markdown(f'<div class="score-grid">{cards}</div>', unsafe_allow_html=True)


def demo_banner(metadata):
    if metadata.get("demo_mode"):
        st.markdown('<div class="notice"><div class="notice-icon">!</div><div><b>DEMO DATA</b> · Results are for software demonstration and are not experimental research findings.</div></div>', unsafe_allow_html=True)


def run_local_explanations(artifacts, row, top_n=10, lime_samples=900):
    shap_table, shap_runtime = artifacts["shap_engine"].explain_local(row, top_n)
    lime_table, lime_runtime = artifacts["lime_engine"].explain_local(row, top_n, lime_samples)
    return shap_table, shap_runtime, lime_table, lime_runtime


initialize_database()
artifacts, frame, dataset_info = get_context()
metadata, metrics = artifacts["metadata"], artifacts["metrics"]

st.sidebar.markdown(
    '''<div class="brand-card"><div class="brand-row"><div class="brand-mark">⬡</div>
    <div><div class="brand-name">E-XAI</div><div class="section-tag">DEFENSE INTELLIGENCE</div></div></div>
    <div class="brand-copy">Explainability assurance for black-box network intrusion detection.</div></div>
    <div class="nav-label">COMMAND CENTER</div>''',
    unsafe_allow_html=True,
)
pages = ["Overview", "Dataset", "Model Performance", "Intrusion Detection", "Explain Prediction",
         "XAI Evaluation", "Framework Comparison", "Global Analysis", "Experiment History", "About / Methodology"]
page = st.sidebar.radio("Navigate", pages, label_visibility="collapsed")
st.sidebar.markdown(
    f'''<div class="model-card"><div class="online"><span class="status-dot"></span> MODEL ONLINE</div>
    <div class="model-name">{metadata["model_name"]}</div>
    <div class="model-meta">{metadata["model_feature_count"]} features · trained {metadata["trained_at"][:10]}</div></div>''',
    unsafe_allow_html=True,
)

page_copy = {
    "Overview": "Live posture, model readiness, and the full explanation assurance pipeline.",
    "Dataset": "Inspect provenance, class balance, data quality, and preprocessing controls.",
    "Model Performance": "Evaluate detection strength beyond accuracy with security-critical metrics.",
    "Intrusion Detection": "Triage a network flow with real model probabilities and risk thresholds.",
    "Explain Prediction": "Interrogate the selected verdict through independent local explainers.",
    "XAI Evaluation": "Stress-test explanation fidelity, stability, consistency, and efficiency.",
    "Framework Comparison": "Compare measured explainer quality without a predetermined winner.",
    "Global Analysis": "Discover model-wide behavior using bounded SHAP and permutation analysis.",
    "Experiment History": "Review, filter, export, and audit stored explanation experiments.",
    "About / Methodology": "Operational definitions, research boundaries, and scientific limitations.",
}
mode_label = "DEMO MODE" if metadata.get("demo_mode") else "RESEARCH DATA"
st.markdown(
    f'''<div class="hero"><div class="eyebrow">SECURITY OPERATIONS / {page.upper()}</div>
    <h1>{page}</h1><div class="hero-copy">{page_copy[page]}</div>
    <div class="hero-meta"><span class="hero-chip live">● SYSTEM READY</span>
    <span class="hero-chip">{metadata["model_name"]}</span><span class="hero-chip">{mode_label}</span>
    <span class="hero-chip">{metadata["samples"]:,} FLOWS</span></div></div>''',
    unsafe_allow_html=True,
)
demo_banner(metadata)

if page == "Overview":
    section_header("Operational snapshot", "Current dataset coverage and black-box detection posture.", "LIVE TELEMETRY")
    kpis([
        ("Dataset Samples", f"{metadata['samples']:,}"), ("Features", str(metadata["model_feature_count"])),
        ("Benign Traffic", f"{metadata['class_distribution']['BENIGN']:,}"),
        ("Attack Traffic", f"{metadata['class_distribution']['ATTACK']:,}"),
        ("Accuracy", f"{metrics['accuracy']:.1%}"), ("F1", f"{metrics['f1']:.1%}"),
    ])
    section_header("Assurance pipeline", "From raw flow telemetry to an auditable explainer comparison.", "SYSTEM FLOW")
    pipeline_panel()
    left, right = st.columns([1, 1.35])
    with left:
        section_header("Traffic composition", "Observed binary class distribution.", "DATASET")
        distribution = pd.DataFrame({"Class": ["BENIGN", "ATTACK"], "Samples": list(metadata["class_distribution"].values())})
        distribution_figure = px.pie(
            distribution, names="Class", values="Samples", hole=.68,
            color="Class", color_discrete_map={"BENIGN": "#5eead4", "ATTACK": "#fb7185"},
        )
        distribution_figure.update_traces(textposition="outside", textinfo="label+percent", marker_line_width=0)
        distribution_figure.update_layout(showlegend=False, height=390, margin={"l":20,"r":20,"t":30,"b":20})
        st.plotly_chart(distribution_figure, width="stretch")
    with right:
        section_header("Recent experiments", "Latest persisted explanation-quality runs.", "AUDIT TRAIL")
        history = read_experiments()
        if history.empty:
            st.info("No saved explanation experiment yet. Use XAI Evaluation to create one.")
        else:
            st.dataframe(history[["created_at", "sample_id", "prediction", "explainer", "quality"]].head(4), hide_index=True, width="stretch")

elif page == "Dataset":
    section_header("Dataset profile", "Provenance, structural quality, and class composition before modeling.", "DATA GOVERNANCE")
    label_column = artifacts["preprocessor"].label_column
    labels = map_binary_labels(normalize_columns(frame)[label_column]) if label_column in normalize_columns(frame).columns else pd.Series(dtype=int)
    kpis([("Source", dataset_info["mode"].upper()), ("Rows", f"{len(frame):,}"), ("Columns", str(frame.shape[1])),
          ("Missing cells", f"{int(frame.isna().sum().sum()):,}"), ("Duplicates", f"{int(frame.duplicated().sum()):,}")])
    if not labels.empty:
        counts = labels.map({0: "BENIGN", 1: "ATTACK"}).value_counts().rename_axis("class").reset_index(name="count")
        class_figure = px.bar(counts, x="class", y="count", color="class", color_discrete_map={"BENIGN":"#5eead4","ATTACK":"#fb7185"})
        class_figure.update_traces(marker_line_width=0)
        class_figure.update_layout(showlegend=False, height=350, xaxis_title=None, yaxis_title="Flows")
        st.plotly_chart(class_figure, width="stretch")
    section_header("Flow preview", "A bounded sample of normalized source records.", "DATA INSPECTION")
    st.dataframe(frame.head(50), width="stretch")
    with st.expander("Preprocessing summary", expanded=True):
        st.json(metadata["preprocessing"])
    section_header("Bring your own dataset", "Upload a compatible labeled CSV to validate, train, and activate it.", "INGESTION")
    upload = st.file_uploader("Upload a compatible network-flow CSV", type=["csv"])
    if upload is not None:
        try:
            uploaded = load_csv(upload)
            st.success(f"Validated {len(uploaded):,} rows and {uploaded.shape[1]} columns.")
            st.dataframe(uploaded.head(), width="stretch")
            if st.button("Train and activate uploaded dataset", type="primary"):
                with st.spinner("Training model and saving artifacts…"):
                    info = {"mode": "upload", "name": upload.name, "files": [upload.name]}
                    result = train_intrusion_model(uploaded, info)
                    save_training_result(result)
                    st.session_state.active_frame = uploaded
                    st.session_state.dataset_info = info
                    for key in ("selected_row", "selected_raw", "prediction", "evaluation"):
                        st.session_state.pop(key, None)
                    st.cache_resource.clear(); st.cache_data.clear()
                st.success("Uploaded dataset trained and activated.")
                st.rerun()
        except Exception as exc:
            st.error(f"CSV validation failed: {exc}")

elif page == "Model Performance":
    section_header("Detection scorecard", "Holdout metrics calculated from model outputs—not hard-coded values.", "MODEL ASSURANCE")
    kpis([(name.replace("_", " ").title(), f"{metrics[name]:.3f}") for name in ("accuracy", "precision", "recall", "f1", "roc_auc")])
    section_header("Diagnostic views", "Class-level errors and discrimination across operating thresholds.", "PERFORMANCE")
    left, right = st.columns(2)
    left.plotly_chart(confusion_matrix_chart(metrics["confusion_matrix"]), width="stretch")
    roc = metrics["roc_curve"]
    roc_figure = px.area(x=roc["fpr"], y=roc["tpr"], labels={"x":"False positive rate", "y":"True positive rate"},
                         title=f"ROC curve · AUC {metrics['roc_auc']:.3f}")
    roc_figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash":"dash"})
    right.plotly_chart(roc_figure, width="stretch")
    section_header("Classification report", "Per-class precision, recall, F1, and support.", "DETAILED METRICS")
    st.dataframe(pd.DataFrame(metrics["classification_report"]).T, width="stretch")
    st.info("Recall measures detected attacks and is critical when missed intrusions are costly; F1 balances that detection rate against false alerts.")

elif page == "Intrusion Detection":
    section_header("Event acquisition", "Select an observed flow or compose a controlled manual scenario.", "TRIAGE")
    mode = st.radio("Input method", ["Select dataset sample", "Manually edit feature values"], horizontal=True)
    if mode == "Select dataset sample":
        index = st.number_input("Dataset row", 0, max(0, len(frame)-1), int(st.session_state.get("selected_index", 0)))
        raw = frame.iloc[[int(index)]].copy()
    else:
        base_index = int(st.session_state.get("selected_index", 0))
        raw = frame.iloc[[base_index]].copy()
        input_columns = artifacts["preprocessor"].input_columns
        st.caption("Common fields are shown first; use Advanced fields for the remainder.")
        for column in input_columns[:8]:
            value = raw.iloc[0][column]
            if pd.api.types.is_number(value):
                raw.loc[raw.index[0], column] = st.number_input(column, value=float(value), key=f"manual_{column}")
            else:
                raw.loc[raw.index[0], column] = st.text_input(column, value=str(value), key=f"manual_{column}")
        with st.expander("Advanced fields"):
            for column in input_columns[8:]:
                value = raw.iloc[0][column]
                if pd.api.types.is_number(value):
                    raw.loc[raw.index[0], column] = st.number_input(column, value=float(value), key=f"manual_{column}")
                else:
                    raw.loc[raw.index[0], column] = st.text_input(column, value=str(value), key=f"manual_{column}")
    if st.button("ANALYZE NETWORK EVENT", type="primary", width="stretch"):
        try:
            transformed = artifacts["preprocessor"].transform(raw)[0]
            st.session_state.selected_row = transformed
            st.session_state.selected_raw = raw
            st.session_state.sample_id = f"dataset-{int(index)}" if mode.startswith("Select") else "manual"
            st.session_state.prediction = predict_transformed(artifacts["model"], transformed)
            st.session_state.selected_index = int(index) if mode.startswith("Select") else base_index
        except Exception as exc:
            st.error(f"Unable to analyze values: {exc}")
    row, prediction = ensure_selected(artifacts, frame)
    section_header("Decision output", "Probability-aware classification generated by the active model.", "ANALYSIS")
    prediction_panel(prediction)
    st.caption("Risk thresholds: LOW <35%, MEDIUM 35–60%, HIGH 60–85%, CRITICAL ≥85%.")
    section_header("Event characteristics", "The major source fields associated with the selected event.", "FLOW CONTEXT")
    st.dataframe(st.session_state.selected_raw.T.rename(columns={st.session_state.selected_raw.index[0]: "value"}).head(15), width="stretch")

elif page == "Explain Prediction":
    row, prediction = ensure_selected(artifacts, frame)
    section_header("Selected verdict", "The same network event is explained independently by SHAP and LIME.", "LOCAL EXPLANATION")
    kpis([("Prediction", prediction["label"]), ("Attack probability", f"{prediction['attack_probability']:.2%}"), ("Risk", prediction["risk"])])
    with st.spinner("Generating real SHAP and LIME explanations…"):
        shap_table, shap_runtime, lime_table, lime_runtime = run_local_explanations(artifacts, row)
    shap_tab, lime_tab = st.tabs([f"SHAP · {shap_runtime:.3f}s", f"LIME · {lime_runtime:.3f}s"])
    with shap_tab:
        st.plotly_chart(contribution_chart(shap_table, "Local SHAP contributions"), width="stretch")
        st.dataframe(shap_table, hide_index=True, width="stretch")
    with lime_tab:
        st.plotly_chart(contribution_chart(lime_table, "Local LIME contributions"), width="stretch")
        st.dataframe(lime_table, hide_index=True, width="stretch")
    leaders = shap_table.head(3)
    description = ", ".join(f"{row.feature} ({row.direction.lower()})" for row in leaders.itertuples())
    st.info(f"For this event, the model decision is most associated with {description}. These are model influences, not evidence of attacker intent or causality.")

elif page == "XAI Evaluation":
    row, prediction = ensure_selected(artifacts, frame)
    section_header("Evaluation controls", "Configure a bounded reliability stress test and analytical score weights.", "EXPERIMENT DESIGN")
    with st.form("evaluation_settings"):
        a, b, c = st.columns(3)
        top_k = a.slider("Top-K", 3, min(15, len(artifacts["feature_names"])), 8)
        perturbations = b.slider("Perturbations", 2, 8, 4)
        strength = c.slider("Perturbation strength", 0.01, 0.10, 0.03, 0.01)
        st.caption("Quality-score weights are configurable analytical choices and do not represent a universal XAI evaluation standard.")
        weight_values = {}
        defaults = {"fidelity":.30, "stability":.25, "consistency":.20, "sparsity":.15, "efficiency":.10}
        weight_cols = st.columns(5)
        for col, (name, default) in zip(weight_cols, defaults.items()):
            weight_values[name] = col.number_input(name.title(), 0.0, 1.0, default, .05)
        submitted = st.form_submit_button("RUN E-XAI EVALUATION", type="primary", width="stretch")
    if submitted:
        if not np.isclose(sum(weight_values.values()), 1.0, atol=1e-6):
            st.error(f"Weights must total 1.00 (current total: {sum(weight_values.values()):.2f}).")
        else:
            with st.status("Evaluating explanation reliability…", expanded=True) as status:
                st.write("Generating local explanations")
                shap_table, _, lime_table, _ = run_local_explanations(artifacts, row, max(10, top_k), 800)
                train = artifacts["train_reference"]["X"]
                st.write("Measuring SHAP perturbation fidelity, stability, and consistency")
                shap_result, _ = evaluate_explainer("SHAP", artifacts["model"], row, artifacts["shap_engine"].explain_vector,
                                                     train, perturbations, strength, top_k, weight_values)
                st.write("Measuring LIME perturbation fidelity, stability, and consistency")
                lime_callable = lambda x: artifacts["lime_engine"].explain_vector(x, 450)
                lime_result, _ = evaluate_explainer("LIME", artifacts["model"], row, lime_callable,
                                                     train, perturbations, strength, top_k, weight_values)
                agreement = top_k_agreement(shap_table, lime_table, top_k)
                for result, table in ((shap_result, shap_table), (lime_result, lime_table)):
                    result["top_features"] = table["feature"].head(top_k).tolist()
                record = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset": dataset_info["name"],
                          "model": metadata["model_name"], "sample_id": st.session_state.sample_id,
                          "prediction": prediction["label"], "attack_probability": prediction["attack_probability"],
                          "risk": prediction["risk"], "agreement_jaccard": agreement["jaccard"],
                          "agreement_spearman": agreement["rank_correlation"], "details": {"weights": weight_values,
                          "top_k": top_k, "perturbations": perturbations, "strength": strength}, "demo_mode": metadata.get("demo_mode")}
                experiment_id = save_experiment(record, [shap_result, lime_result])
                st.session_state.evaluation = {"results":[shap_result, lime_result], "agreement":agreement,
                                               "shap":shap_table, "lime":lime_table, "record":record, "id":experiment_id}
                status.update(label=f"Evaluation complete · experiment {experiment_id}", state="complete")
    evaluation = st.session_state.get("evaluation")
    if evaluation:
        result_frame = pd.DataFrame(evaluation["results"])
        section_header("Measured quality", "Higher values indicate stronger performance under this configured operational definition.", "RESULTS")
        evaluation_cards(evaluation["results"])
        st.dataframe(result_frame[["explainer","fidelity","stability","consistency","sparsity","runtime","efficiency","quality"]], hide_index=True, width="stretch")
        st.plotly_chart(radar_chart(evaluation["results"]), width="stretch")
        st.caption("Fidelity is cosine agreement with one-feature-at-a-time probability sensitivity. Stability compares prediction-preserving perturbations. Consistency compares same-class nearest neighbors. Sparsity is 1 − significant features / total features, using 10% of maximum weight. Efficiency is 1/(1+seconds). These are operational project definitions, not universal standards.")

elif page == "Framework Comparison":
    evaluation = st.session_state.get("evaluation")
    if not evaluation:
        st.info("Run an E-XAI evaluation first. Saved historical results are available under Experiment History.")
    else:
        result_frame = pd.DataFrame(evaluation["results"])
        winner = result_frame.loc[result_frame["quality"].idxmax(), "explainer"]
        section_header("Empirical outcome", "The leading framework is determined only by the current measured experiment.", "HEAD-TO-HEAD")
        st.success(f"Highest measured quality for this experiment: {winner}")
        evaluation_cards(evaluation["results"])
        st.dataframe(result_frame[["explainer","fidelity","stability","consistency","sparsity","runtime","quality"]], hide_index=True, width="stretch")
        agreement = evaluation["agreement"]
        kpis([("Top-K Jaccard", f"{agreement['jaccard']:.3f}"), ("Rank correlation", f"{agreement['rank_correlation']:.3f}"),
              ("Common features", str(len(agreement["common_features"])))])
        st.write("Common Top-K features:", ", ".join(agreement["common_features"]) or "None")
        comparison = pd.merge(evaluation["shap"][["feature","importance"]], evaluation["lime"][["feature","importance"]], on="feature", how="outer", suffixes=("_shap","_lime")).fillna(0)
        st.plotly_chart(px.bar(comparison.head(15), x="feature", y=["importance_shap","importance_lime"], barmode="group", title="Local ranking comparison"), width="stretch")
        report = generate_html_report(evaluation["record"], evaluation["results"], agreement, metrics)
        st.download_button("Download HTML experiment report", report, f"exai_experiment_{evaluation['id']}.html", "text/html")

elif page == "Global Analysis":
    section_header("Model-wide intelligence", "Bounded global rankings reveal recurring model behavior across the reference set.", "GLOBAL XAI")
    st.warning("Global importance summarizes model behavior across samples; it does not explain an individual prediction.")
    if st.button("CALCULATE GLOBAL IMPORTANCE", type="primary") or "global_analysis" in st.session_state:
        if "global_analysis" not in st.session_state:
            with st.spinner("Calculating bounded global SHAP and permutation importance…"):
                X, y = artifacts["test_reference"]["X"], artifacts["test_reference"]["y"]
                shap_global, shap_time = artifacts["shap_engine"].global_importance(X, 150)
                permutation, permutation_time = calculate_permutation_importance(artifacts["model"], X, y, artifacts["feature_names"], 4)
                st.session_state.global_analysis = (shap_global, shap_time, permutation, permutation_time)
        shap_global, shap_time, permutation, permutation_time = st.session_state.global_analysis
        left, right = st.columns(2)
        left.plotly_chart(importance_chart(shap_global, "mean_abs_shap", f"Global SHAP · {shap_time:.3f}s"), width="stretch")
        right.plotly_chart(importance_chart(permutation, "importance_mean", f"Permutation F1 importance · {permutation_time:.3f}s"), width="stretch")
        if hasattr(artifacts["model"], "feature_importances_"):
            model_table = pd.DataFrame({"feature":artifacts["feature_names"], "model_importance":artifacts["model"].feature_importances_}).sort_values("model_importance", ascending=False)
            st.plotly_chart(importance_chart(model_table, "model_importance", "Model-native feature importance"), width="stretch")

elif page == "Experiment History":
    section_header("Experiment ledger", "Persistent, filterable evidence for comparison and reproducibility.", "SQLITE AUDIT")
    history = read_experiments()
    if history.empty:
        st.info("No experiments saved yet.")
    else:
        choices = ["All"] + sorted(history["explainer"].dropna().unique().tolist())
        selected = st.selectbox("Explainer filter", choices)
        view = history if selected == "All" else history[history["explainer"] == selected]
        display = view.copy()
        display["created_at"] = pd.to_datetime(display["created_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        display["dataset"] = (
            display["dataset"].astype(str)
            .replace("Deterministic network-flow demonstration data", "Demo network flows")
            .str.slice(0, 20)
        )
        display = display[[
            "id", "created_at", "dataset", "sample_id", "prediction", "attack_probability",
            "risk", "explainer", "quality", "runtime",
        ]]
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "id": st.column_config.NumberColumn("Run", format="%d", width="small"),
                "created_at": st.column_config.TextColumn("Timestamp", width="medium"),
                "dataset": st.column_config.TextColumn("Dataset", width="small"),
                "sample_id": st.column_config.TextColumn("Sample", width="small"),
                "prediction": st.column_config.TextColumn("Verdict", width="small"),
                "risk": st.column_config.TextColumn("Risk", width="small"),
                "explainer": st.column_config.TextColumn("Explainer", width="small"),
                "attack_probability": st.column_config.ProgressColumn("Attack probability", min_value=0.0, max_value=1.0, format="%.3f", width="small"),
                "quality": st.column_config.ProgressColumn("Quality", min_value=0.0, max_value=1.0, format="%.3f", width="small"),
                "runtime": st.column_config.NumberColumn("Runtime (s)", format="%.3f", width="small"),
            },
        )
        st.download_button("Export history as CSV", view.to_csv(index=False), "exai_history.csv", "text/csv")
        with st.expander("Inspect experiment details"):
            experiment_id = st.selectbox("Experiment ID", sorted(history["id"].unique(), reverse=True))
            st.json(history[history["id"] == experiment_id].to_dict(orient="records"))
        confirm = st.checkbox("I understand this clears experiment history")
        if st.button("Clear history", disabled=not confirm):
            clear_history(); st.success("Experiment history cleared."); st.rerun()

else:
    section_header(
        "Research question",
        "How reliable are different black-box XAI frameworks when explaining network-intrusion decisions?",
        "E-XAI METHODOLOGY",
    )
    st.markdown(
        '''<div class="method-grid">
        <div class="method-card"><div class="method-icon">F</div><div class="method-title">Fidelity</div><div class="method-copy">Cosine agreement between explanation magnitude and controlled one-feature masking sensitivity.</div></div>
        <div class="method-card"><div class="method-icon">S</div><div class="method-title">Stability</div><div class="method-copy">Top-K Jaccard and cosine similarity under small, prediction-preserving perturbations.</div></div>
        <div class="method-card"><div class="method-icon">C</div><div class="method-title">Consistency</div><div class="method-copy">Explanation similarity among nearest reference events with the same predicted class.</div></div>
        <div class="method-card"><div class="method-icon">SP</div><div class="method-title">Sparsity</div><div class="method-copy">One minus the proportion of features reaching 10% of the largest absolute contribution.</div></div>
        <div class="method-card"><div class="method-icon">T</div><div class="method-title">Runtime</div><div class="method-copy">Measured wall-clock generation time with efficiency normalized as 1 / (1 + seconds).</div></div>
        <div class="method-card"><div class="method-icon">A</div><div class="method-title">Agreement</div><div class="method-copy">SHAP/LIME Top-K Jaccard overlap and Spearman correlation among common ranks.</div></div>
        </div>''',
        unsafe_allow_html=True,
    )
    section_header("Composite quality", "A configurable analytical summary—not a universal XAI standard.", "QUALITY MODEL")
    st.code("Q = 0.30F + 0.25S + 0.20C + 0.15SP + 0.10E", language=None)
    section_header("Scientific boundaries", "Interpret every result within the model, dataset, and metric design used.", "LIMITATIONS")
    st.warning(
        "Explanation quality depends on the underlying model and dataset. LIME varies with perturbation sampling; "
        "SHAP can be expensive; offline datasets do not prove real-time performance. Agreement does not establish "
        "causal correctness. Synthetic demonstration data is never presented as research evidence."
    )
