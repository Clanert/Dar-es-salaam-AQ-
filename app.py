# app.py
# Dar es Salaam Air Quality Dashboard (Streamlit)
# Upgrade: pollutant selector (PM1, PM2.5, PM10, NO2, O3) + map layers with collapsible control.

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster, Fullscreen
from branca.colormap import linear, LinearColormap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from urllib.request import urlopen
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Dar es Salaam Air Quality",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Constants & Utilities
# ------------------------------
WHO_PM25_24H = 15   # µg/m³
WHO_PM10_24H = 45   # µg/m³
# (Optional) WHO references for NO2/O3 left out to avoid confusion over averaging method (24h vs 8h).

PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200),
    (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500),
]

AQI_LABELS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

POLLUTANT_COLUMNS: Dict[str, str] = {
    "PM1": "pm1",
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "O3": "o3",
}
POLLUTANT_UNITS: Dict[str, str] = {k: "µg/m³" for k in POLLUTANT_COLUMNS}

@dataclass
class Station:
    id: str
    name: str
    lat: float
    lon: float

# ------------------------------
# Report helpers (rich version)
# ------------------------------
def _aqi_color(aqi: float) -> str:
    for lo, hi, _, color in AQI_LABELS:
        if lo <= aqi <= hi:
            return color
    return "#7e0023"

def _aqi_legend_table():
    data = [["Category", "Range", "Colour"]]
    for lo, hi, label, color in AQI_LABELS:
        data.append([label, f"{lo} – {hi}", ""])
    tbl = Table(data, colWidths=[2.4*inch, 1.6*inch, 1.0*inch])
    styles = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.black),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
    ]
    for i, (_, _, _, color) in enumerate(AQI_LABELS, start=1):
        styles.append(("BACKGROUND", (2,i), (2,i), colors.HexColor(color)))
    tbl.setStyle(TableStyle(styles))
    return tbl

def _try_fetch_logo(logo_url: str, width_px: int = 160) -> Image | None:
    try:
        raw = urlopen(logo_url, timeout=10).read()
        bio = BytesIO(raw)
        img = Image(bio, width=width_px, height=width_px*(44/160))
        return img
    except Exception:
        return None

def _color_for_aqi(aqi: float) -> str:
    for low, high, _, color in AQI_LABELS:
        if low <= aqi <= high:
            return color
    return "#7e0023"

def _label_for_aqi(aqi: float) -> str:
    for low, high, label, _ in AQI_LABELS:
        if low <= aqi <= high:
            return label
    return "Hazardous"

def _linear_scale(cp: float, bp_lo: float, bp_hi: float, i_lo: float, i_hi: float) -> float:
    return (i_hi - i_lo) / (bp_hi - bp_lo) * (cp - bp_lo) + i_lo

def _aqi_from_breakpoints(cp: float, bps: list[Tuple[float, float, int, int]]) -> float:
    for bp_lo, bp_hi, i_lo, i_hi in bps:
        if bp_lo <= cp <= bp_hi:
            return round(_linear_scale(cp, bp_lo, bp_hi, i_lo, i_hi))
    bp_lo, bp_hi, i_lo, i_hi = bps[-1]
    if cp > bp_hi:
        return round(_linear_scale(cp, bp_lo, bp_hi, i_lo, i_hi))
    return np.nan

def aqi_from_pm(pm25: float | pd.Series, pm10: float | pd.Series):
    pm25_aqi = pd.Series(pm25).apply(lambda x: _aqi_from_breakpoints(x, PM25_BREAKPOINTS))
    pm10_aqi = pd.Series(pm10).apply(lambda x: _aqi_from_breakpoints(x, PM10_BREAKPOINTS))
    overall = pd.concat([pm25_aqi, pm10_aqi], axis=1).max(axis=1)
    return pm25_aqi, pm10_aqi, overall

def generate_report(df: pd.DataFrame, stations: List[str], start: datetime, end: datetime) -> bytes:
    """Rich PDF report (kept)."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, rightMargin=36, leftMargin=36, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], leading=14)
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=9, textColor=colors.grey)

    story = []
    logo = _try_fetch_logo("https://i.ibb.co/gLc9tqzN/download.jpg", width_px=180)
    if logo:
        header_tbl = Table(
            [[logo, Paragraph("<b>Dar es Salaam Air Quality Report</b>", h1)]],
            colWidths=[1.8*inch, None]
        )
        header_tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
        story.append(header_tbl)
    else:
        story.append(Paragraph("Dar es Salaam Air Quality Report", h1))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Period: <b>{start.strftime('%Y-%m-%d')}</b> to <b>{end.strftime('%Y-%m-%d')}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Stations: <b>{', '.join([s for s in stations if s!='All']) if stations and 'All' not in stations else 'All'}</b>",
        small
    ))
    story.append(HRFlowable(color=colors.HexColor("#e0e0e0"), thickness=0.8, spaceBefore=6, spaceAfter=10))
    story.append(Paragraph("About this report", h2))
    story.append(Paragraph(
        "Air quality is summarized using the Air Quality Index (AQI), translated from PM₂.₅ and PM₁₀. "
        "Categories: Good (0–50), Moderate (51–100), USG (101–150), Unhealthy (151–200), Very Unhealthy (201–300), Hazardous (301–500).",
        body
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("AQI Categories (Colour-coded)", h2))
    story.append(_aqi_legend_table())
    story.append(Spacer(1, 10))

    pm25_mean = round(df["pm25"].mean(), 1) if "pm25" in df else float("nan")
    pm10_mean = round(df["pm10"].mean(), 1) if "pm10" in df else float("nan")
    pm25_aqi, pm10_aqi, overall = aqi_from_pm(df.get("pm25", pd.Series(dtype=float)), df.get("pm10", pd.Series(dtype=float)))
    df_stats = df.copy()
    df_stats["AQI"] = overall.values
    city_mean_aqi = round(df_stats["AQI"].mean(), 1) if not df_stats.empty else float("nan")
    poor = df_stats.sort_values("AQI", ascending=False).iloc[0]
    best = df_stats.sort_values("AQI", ascending=True).iloc[0]

    kpi_data = [
        ["Citywide mean AQI", str(city_mean_aqi)],
        ["Citywide mean PM₂.₅ (µg/m³)", str(pm25_mean)],
        ["Citywide mean PM₁₀ (µg/m³)", str(pm10_mean)],
        [f"poor: {poor['station_name']} ({poor['station_id']})", f"AQI {int(poor['AQI'])}"],
        [f"Best: {best['station_name']} ({best['station_id']})", f"AQI {int(best['AQI'])}"],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[3.4*inch, None])
    kpi_style = [
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#888")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f7f7f7")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]
    kpi_style.append(("BACKGROUND", (1,3), (1,3), colors.HexColor(_aqi_color(float(poor["AQI"])))))
    kpi_style.append(("BACKGROUND", (1,4), (1,4), colors.HexColor(_aqi_color(float(best["AQI"])))))
    kpi_tbl.setStyle(TableStyle(kpi_style))
    story.append(kpi_tbl)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Per-station averages (selected period)", h2))
    agg = df_stats.groupby(["station_name","station_id"])[["pm25","pm10","AQI"]].mean().round(1).reset_index()
    data = [["Station", "PM₂.₅", "PM₁₀", "AQI"]]
    for r in agg.itertuples(index=False):
        data.append([f"{r.station_name} ({r.station_id})", r.pm25, r.pm10, r.AQI])
    tbl = Table(data, colWidths=[3.6*inch, 1.0*inch, 1.0*inch, 0.9*inch])
    stl = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#333333")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("ALIGN",      (1,1), (-1,-1), "CENTER"),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
        ("GRID",       (0,0), (-1,-1), 0.25, colors.HexColor("#888")),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
    ]
    for i in range(1, len(data)):
        aqi_val = data[i][3]
        stl.append(("BACKGROUND", (3,i), (3,i), colors.HexColor(_aqi_color(float(aqi_val)))))
    tbl.setStyle(TableStyle(stl))
    story.append(tbl)
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Note: AQI computed from PM₂.₅ and PM₁₀ using U.S. EPA breakpoints. WHO 2021 24-hour guidelines: PM₂.₅ = 15 µg/m³, PM₁₀ = 45 µg/m³.",
        ParagraphStyle("Small2", parent=small)
    ))
    doc.build(story)
    buf.seek(0)
    return buf.read()

# Legacy basic report kept but renamed to avoid override (not used).
def generate_report_basic(df: pd.DataFrame, stations: list[str], start: datetime, end: datetime) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Dar es Salaam Air Quality Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}", styles["Normal"]))
    story.append(Paragraph(f"Stations: {', '.join(stations) if stations else 'All'}", styles["Normal"]))
    story.append(Spacer(1, 12))
    pm25_mean = round(df["pm25"].mean(), 1) if "pm25" in df else float("nan")
    pm10_mean = round(df["pm10"].mean(), 1) if "pm10" in df else float("nan")
    story.append(Paragraph(f"Citywide mean PM₂.₅: {pm25_mean} µg/m³", styles["Normal"]))
    story.append(Paragraph(f"Citywide mean PM₁₀: {pm10_mean} µg/m³", styles["Normal"]))
    pm25_aqi, pm10_aqi, overall = aqi_from_pm(df.get("pm25", pd.Series(dtype=float)), df.get("pm10", pd.Series(dtype=float)))
    df = df.copy(); df["AQI"] = overall.values
    poor = df.sort_values("AQI", ascending=False).iloc[0]
    story.append(Paragraph(
        f"poor station in this period: {poor['station_name']} ({poor['station_id']}) with AQI {int(poor['AQI'])} at {poor['timestamp']}", styles["Normal"]
    ))
    agg = df.groupby("station_name")[["pm25","pm10","AQI"]].mean().round(1).reset_index()
    data = [["Station", "PM₂.₅", "PM₁₀", "AQI"]] + agg.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ]))
    story.append(table)
    doc.build(story)
    buf.seek(0)
    return buf.read()

# ------------------------------
# Data loading / simulation
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    def _ensure_pollutants(df_in: pd.DataFrame) -> pd.DataFrame:
        df_o = df_in.copy()
        if "pm1" not in df_o.columns:
            # Why: Provide PM1 when not recorded, proportional to PM2.5 with noise.
            base = df_o.get("pm25", pd.Series(12, index=df_o.index))
            df_o["pm1"] = np.round(np.clip(base * np.random.uniform(0.55, 0.85), 2, 120), 1)
        if "no2" not in df_o.columns:
            # Diurnal NO2 with morning/evening peaks
            t = pd.to_datetime(df_o["timestamp"])
            hour = t.dt.hour.to_numpy() if not t.empty else np.array([0])
            diurnal = 25 + 15*np.exp(-((hour-7)/3)**2) + 12*np.exp(-((hour-19)/3)**2)
            noise = np.random.normal(0, 4, size=len(df_o))
            df_o["no2"] = np.round(np.clip(diurnal + noise, 5, 120), 1)
        if "o3" not in df_o.columns:
            # Midday O3 peak, some anti-correlation with NO2
            t = pd.to_datetime(df_o["timestamp"])
            hour = t.dt.hour.to_numpy() if not t.empty else np.array([0])
            diurnal = 20 + 28*np.exp(-((hour-13)/3.5)**2)
            noise = np.random.normal(0, 5, size=len(df_o))
            o3_raw = np.clip(diurnal + noise, 5, 180)
            # Light anti-correlation if NO2 exists
            if "no2" in df_o:
                o3_raw = o3_raw - 0.08*(df_o["no2"].to_numpy() - df_o["no2"].mean())
            df_o["o3"] = np.round(np.clip(o3_raw, 5, 180), 1)
        return df_o

    if os.path.exists("dsm_air_quality.csv"):
        df = pd.read_csv("dsm_air_quality.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = _ensure_pollutants(df)
        return df

    # Simulated dataset
    np.random.seed(42)
    base_stations = [
        ("DSM01", "Kivukoni", -6.815, 39.292), ("DSM02", "Upanga", -6.809, 39.279), ("DSM03", "Oysterbay", -6.748, 39.279),
        ("DSM04", "Masaki", -6.734, 39.268), ("DSM05", "Mikocheni", -6.761, 39.238), ("DSM06", "Kinondoni", -6.777, 39.249),
        ("DSM07", "Ubungo", -6.774, 39.203), ("DSM08", "Kimara", -6.736, 39.166), ("DSM09", "Temeke", -6.857, 39.218),
        ("DSM10", "Chang'ombe", -6.861, 39.245), ("DSM11", "Mbagala", -6.910, 39.271), ("DSM12", "Kigamboni", -6.877, 39.311),
        ("DSM13", "Kipawa", -6.848, 39.219), ("DSM14", "Tabata", -6.834, 39.249), ("DSM15", "Kawe", -6.739, 39.224),
    ]
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=14)
    idx = pd.date_range(start, end, freq="H", tz="UTC")

    rows = []
    for sid, name, lat, lon in base_stations:
        pm25_base = np.clip(10 + 10*np.sin(np.linspace(0, 12*np.pi, len(idx))) + np.random.normal(0, 6, len(idx)), 4, 150)
        pm10_base = pm25_base * np.random.uniform(1.3, 2.0)
        temp = np.clip(26 + 3*np.sin(np.linspace(0, 7*np.pi, len(idx))) + np.random.normal(0, 1.6, len(idx)), 22, 35)
        rh = np.clip(72 + 10*np.cos(np.linspace(0, 7*np.pi, len(idx))) + np.random.normal(0, 6, len(idx)), 45, 95)
        # spikes
        for spike in np.random.choice(range(0, len(idx)-6, 24), size=3, replace=False):
            pm25_base[spike: spike+3] += np.random.uniform(25, 70)
            pm10_base[spike: spike+3] += np.random.uniform(50, 120)
        # extra pollutants
        pm1 = np.clip(pm25_base * np.random.uniform(0.6, 0.85), 2, 120)
        hour = np.array([t.hour for t in idx])
        no2 = 25 + 15*np.exp(-((hour-7)/3)**2) + 12*np.exp(-((hour-19)/3)**2) + np.random.normal(0, 4, len(idx))
        no2 = np.clip(no2, 5, 120)
        o3 = 20 + 28*np.exp(-((hour-13)/3.5)**2) + np.random.normal(0, 5, len(idx))
        o3 = np.clip(o3 - 0.06*(no2 - np.mean(no2)), 5, 180)

        rows.append(pd.DataFrame({
            "timestamp": idx,
            "station_id": sid,
            "station_name": name,
            "lat": lat,
            "lon": lon,
            "pm1": np.round(pm1, 1),
            "pm25": np.round(pm25_base, 1),
            "pm10": np.round(pm10_base, 1),
            "no2": np.round(no2, 1),
            "o3": np.round(o3, 1),
            "temperature": np.round(temp, 1),
            "humidity": np.round(rh, 0),
        }))
    return pd.concat(rows, ignore_index=True)

# ------------------------------
# UI Helpers & Components
# ------------------------------
def inject_css():
    st.markdown("""
    <style>
      header { visibility: visible !important; }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"] { display: none !important; }
      [data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; z-index: 1000 !important; }
      .block-container{padding-top:1rem;padding-bottom:2rem;}
      footer {visibility: hidden;}
      .kpi-card {border-radius:16px;padding:14px 16px;background:#1112170d;border:1px solid #eaeaea;}
      .brand {display:flex; align-items:center; gap:12px;}
      .brand h1{margin:0;font-size:1.25rem}
      .brand small{opacity:.75}
      .aqi-legend { position:absolute; z-index:9999; bottom:20px; right:20px;
                    background:white; padding:10px 12px; border-radius:10px; border:1px solid #ddd;
                    box-shadow:0 2px 10px rgba(0,0,0,0.1); font-size:13px; }
      .aqi-legend div { display:flex; align-items:center; gap:8px; margin:4px 0; }
      .aqi-legend span { display:inline-block; width:16px; height:10px; border-radius:2px; border:1px solid #6662; }
    </style>
    """, unsafe_allow_html=True)

def top_brand_bar():
    st.markdown(
        """
        <div class="brand">
            <img src="https://i.ibb.co/gLc9tqzN/download.jpg" alt="City Logo" height="44"/>
            <div>
              <h1>Dar es Salaam City Council – Air Quality Intelligence</h1>
              <small>PM₁ • PM₂.₅ • PM₁₀ • NO₂ • O₃ • Temperature • Humidity • AQI</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def station_selector(df: pd.DataFrame) -> list[str]:
    stations = df[["station_id", "station_name"]].drop_duplicates().sort_values("station_name")
    labels = [f"{r.station_name} ({r.station_id})" for r in stations.itertuples(index=False)]
    options = stations.station_id.tolist()
    default = options[:1]
    selected = st.multiselect("Select station(s)", options=options, default=default, format_func=lambda x: labels[options.index(x)])
    return selected

def line_with_threshold(df: pd.DataFrame, y: str, title: str, who_line: float | None = None, yaxis_title: str | None = None, color_col: str | None = None):
    fig = px.line(df, x="timestamp", y=y, color=color_col if color_col and color_col in df.columns else None, title=title)
    if who_line is not None:
        fig.add_hline(y=who_line, line_dash="dash", annotation_text=f"WHO: {who_line} µg/m³", annotation_position="top left")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), yaxis_title=yaxis_title or y)
    st.plotly_chart(fig, use_container_width=True)

def aqi_bar(df: pd.DataFrame, title: str):
    pm25_aqi, pm10_aqi, overall = aqi_from_pm(df["pm25"], df["pm10"])
    d = df[["station_name"]].copy()
    d["AQI"] = overall.values
    d = d.sort_values("AQI", ascending=False)
    fig = px.bar(d, x="station_name", y="AQI", title=title)
    fig.update_traces(marker_color=[_color_for_aqi(a) for a in d["AQI"]])
    fig.update_layout(xaxis_title="Station", yaxis_title="AQI", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def resample_and_classify(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    cols = ["pm1", "pm25", "pm10", "no2", "o3", "temperature", "humidity"]
    cols = [c for c in cols if c in df.columns]
    agg = (df.set_index("timestamp")
             .groupby("station_name")[cols]
             .resample(freq)
             .mean()
             .reset_index())
    if {"pm25","pm10"}.issubset(agg.columns):
        pm25_aqi, pm10_aqi, overall = aqi_from_pm(agg["pm25"], agg["pm10"])
        agg["AQI_PM25"] = pm25_aqi.values
        agg["AQI_PM10"] = pm10_aqi.values
        agg["AQI_overall"] = overall.values
        agg["AQI_category"] = agg["AQI_overall"].apply(_label_for_aqi)
    return agg

def _aqi_legend_html() -> str:
    rows = "".join(
        f'<div><span style="background:{color}"></span><b style="min-width:90px;">{label}</b><i>{lo}-{hi}</i></div>'
        for lo, hi, label, color in AQI_LABELS
    )
    return f'<div class="aqi-legend"><b>AQI Categories</b>{rows}</div>'

def _linear_colormap(vmin: float, vmax: float) -> LinearColormap:
    # Why: Fallback-safe colormap for pollutant layers.
    colorscale = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026']
    return LinearColormap(colorscale, vmin=vmin, vmax=vmax)

def make_map(df_latest: pd.DataFrame, selected_pollutant: str) -> folium.Map:
    if df_latest[["lat","lon"]].dropna().empty:
        center = (-6.8, 39.26)
        m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")
    else:
        south, north = df_latest["lat"].min()-0.1, df_latest["lat"].max()+0.1
        west, east = df_latest["lon"].min()-0.25, df_latest["lon"].max()+0.25
        m = folium.Map(location=[(south+north)/2, (west+east)/2], zoom_start=12, tiles="cartodbpositron")
        m.fit_bounds([[south, west], [north, east]])

    Fullscreen().add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="© Esri, Maxar, Earthstar Geographics, GIS User Community",
        name="Satellite"
    ).add_to(m)

    # AQI markers layer (always on)
    pm25_aqi, pm10_aqi, overall = aqi_from_pm(df_latest["pm25"], df_latest["pm10"])
    df_latest_aqi = df_latest.copy()
    df_latest_aqi["aqi"] = overall.values
    aqi_group = folium.FeatureGroup(name="AQI (PM-based)", show=True)
    cluster = MarkerCluster()
    aqi_group.add_child(cluster)
    for r in df_latest_aqi.itertuples(index=False):
        color = _color_for_aqi(r.aqi)
        html = f"""
        <b>{r.station_name} ({r.station_id})</b><br/>
        AQI: <b>{int(r.aqi)}</b> ({_label_for_aqi(r.aqi)})<br/>
        PM1: {getattr(r, 'pm1', np.nan)} µg/m³<br/>
        PM2.5: {r.pm25} µg/m³ | PM10: {r.pm10} µg/m³<br/>
        NO2: {getattr(r, 'no2', np.nan)} µg/m³ | O3: {getattr(r, 'o3', np.nan)} µg/m³<br/>
        Temp: {r.temperature} °C | RH: {r.humidity}%<br/>
        <small>Updated: {pd.to_datetime(r.timestamp).strftime('%Y-%m-%d %H:%M UTC')}</small>
        """
        folium.CircleMarker(
            location=(r.lat, r.lon), radius=10, weight=1, color="#222",
            fill=True, fill_color=color, fill_opacity=0.92,
            popup=folium.Popup(html, max_width=340),
            tooltip=f"{r.station_name}: AQI {int(r.aqi)} ({_label_for_aqi(r.aqi)})",
        ).add_to(cluster)
    aqi_group.add_to(m)

    # Pollutant layers
    for display_name, col in POLLUTANT_COLUMNS.items():
        if col not in df_latest.columns:
            continue
        grp = folium.FeatureGroup(name=f"{display_name} layer", show=(display_name == selected_pollutant))
        vmin = float(np.nanmin(df_latest[col].values)) if len(df_latest) else 0.0
        vmax = float(np.nanmax(df_latest[col].values)) if len(df_latest) else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, max(1.0, (vmax if np.isfinite(vmax) else 1.0))
        cmap = _linear_colormap(vmin, vmax)
        for r in df_latest.itertuples(index=False):
            val = getattr(r, col)
            fill_col = cmap(val) if np.isfinite(val) else "#888888"
            tip = f"{r.station_name}: {display_name} {val} {POLLUTANT_UNITS[display_name]}"
            html = f"""
            <b>{r.station_name} ({r.station_id})</b><br/>
            {display_name}: <b>{val}</b> {POLLUTANT_UNITS[display_name]}<br/>
            <small>Updated: {pd.to_datetime(r.timestamp).strftime('%Y-%m-%d %H:%M UTC')}</small>
            """
            folium.CircleMarker(
                location=(r.lat, r.lon), radius=9, weight=0.8, color="#333",
                fill=True, fill_color=fill_col, fill_opacity=0.9,
                popup=folium.Popup(html, max_width=280),
                tooltip=tip,
            ).add_to(grp)
        grp.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)  # collapsible layer control
    folium.Map.get_root(m).html.add_child(folium.Element(_aqi_legend_html()))
    return m

# ------------------------------
# Gauges & Alerts
# ------------------------------
def aqi_gauge(aqi_value: float, title: str):
    steps = [{"range": [lo, hi], "color": color} for lo, hi, _, color in AQI_LABELS]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(aqi_value),
        number={"suffix": ""},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 500]},
            "bar": {"color": _color_for_aqi(aqi_value)},
            "steps": steps,
            "threshold": {"line": {"color": _color_for_aqi(aqi_value), "width": 3}, "thickness": 0.9, "value": float(aqi_value)},
        },
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=260)
    st.plotly_chart(fig, use_container_width=True)

def pollution_alert_box(df_window: pd.DataFrame, threshold: int = 151):
    pm25_aqi, pm10_aqi, overall = aqi_from_pm(df_window["pm25"], df_window["pm10"])
    d = df_window[["timestamp", "station_id", "station_name"]].copy()
    d["AQI"] = overall.values
    peaks = (d.sort_values("AQI", ascending=False).groupby("station_id", as_index=False).first())
    offenders = peaks[peaks["AQI"] >= threshold].copy()
    if len(offenders):
        lines = []
        for r in offenders.itertuples(index=False):
            lines.append(f"- **{r.station_name} ({r.station_id})** reached **AQI {int(r.AQI)}** at **{pd.to_datetime(r.timestamp).strftime('%Y-%m-%d %H:%M UTC')}**")
        st.error("🚨 High pollution detected in the last 24 hours:\n\n" + "\n".join(lines))

# ------------------------------
# Sidebar & Navigation
# ------------------------------
with st.sidebar:
    with st.expander("Navigation", expanded=True):
        choice = option_menu(
            "",
            ["Overview", "Stations", "Trends", "Data", "Reports", "About"],
            icons=["globe", "pin-map", "graph-up", "table", "file-earmark-text", "info-circle"],
            default_index=0,
            orientation="vertical",
        )
    st.image("https://i.ibb.co/gLc9tqzN/download.jpg", caption="Dar es Salaam City Council", use_container_width=True)
    st.markdown("---")
    st.markdown("**WHO 24-hour Guidelines**\n\n• PM₂.₅: **15 µg/m³**\n\n• PM₁₀: **45 µg/m³**")
    st.markdown("---")
    selected_pollutant = st.selectbox(
        "Display pollutant",
        list(POLLUTANT_COLUMNS.keys()),
        index=1  # default PM2.5
    )

# Load data
with st.spinner("Loading data…"):
    df = load_data().sort_values("timestamp")

latest_ts = df["timestamp"].max()
df_latest = df[df["timestamp"] == latest_ts]
last_24h_start = latest_ts - pd.Timedelta(hours=24)
df_last24 = df[(df["timestamp"] > last_24h_start) & (df["timestamp"] <= latest_ts)]

# ------------------------------
# KPIs
# ------------------------------
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    st.metric("Stations online", int(df_latest["station_id"].nunique()))
    st.markdown("</div>", unsafe_allow_html=True)
with col_b:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    city_pm25 = round(df_last24["pm25"].mean(),1)
    st.metric("Citywide PM₂.₅ (24h mean)", f"{city_pm25} µg/m³")
    st.markdown("</div>", unsafe_allow_html=True)
with col_c:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    city_pm10 = round(df_last24["pm10"].mean(),1)
    st.metric("Citywide PM₁₀ (24h mean)", f"{city_pm10} µg/m³")
    st.markdown("</div>", unsafe_allow_html=True)
with col_d:
    st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
    poor = df_latest.assign(AQI=aqi_from_pm(df_latest["pm25"], df_latest["pm10"])[2])\
                     .sort_values("AQI", ascending=False).iloc[0]
    st.metric("poor station AQI", f"{int(poor.AQI)}", poor.station_name)
    st.markdown("</div>", unsafe_allow_html=True)

pollution_alert_box(df_last24, threshold=151)

# ------------------------------
# Pages
# ------------------------------
if choice == "Overview":
    st.subheader("Citywide Snapshot")
    # Map with pollutant layers
    m = make_map(df_latest, selected_pollutant)
    st.components.v1.html(m._repr_html_(), height=820, scrolling=False)
    st.caption("Use the layer control (top-right) to expand/collapse and toggle AQI or pollutant layers; use fullscreen for a panoramic view.")

    # Existing AQI + PM plots preserved
    c1, c2 = st.columns([1, 1])
    with c1:
        aqi_bar(df_latest, title=f"AQI by Station (as of {latest_ts.strftime('%Y-%m-%d %H:%M UTC')})")
    with c2:
        city24 = df_last24.groupby("timestamp")[["pm25", "pm10"]].mean().reset_index()
        line_with_threshold(city24, "pm25", "Citywide PM₂.₅ – last 24 hours (mean of stations)", WHO_PM25_24H, "µg/m³")
        line_with_threshold(city24, "pm10", "Citywide PM₁₀ – last 24 hours (mean of stations)", WHO_PM10_24H, "µg/m³")

    # New: selected pollutant citywide line (24h)
    sel_col = POLLUTANT_COLUMNS[selected_pollutant]
    if sel_col in df_last24:
        city24_sel = df_last24.groupby("timestamp")[sel_col].mean().reset_index()
        line_with_threshold(
            city24_sel, sel_col,
            f"Selected pollutant ({selected_pollutant}) – last 24 hours (mean of stations)",
            who_line=WHO_PM25_24H if sel_col=="pm25" else (WHO_PM10_24H if sel_col=="pm10" else None),
            yaxis_title=POLLUTANT_UNITS[selected_pollutant]
        )

    # Gauges (unchanged)
    pm25_aqi_o, pm10_aqi_o, overall_latest = aqi_from_pm(df_latest["pm25"], df_latest["pm10"])
    latest_df_copy = df_latest.copy()
    latest_df_copy["AQI"] = overall_latest.values
    city_mean_aqi = float(latest_df_copy["AQI"].mean())
    poor_row = latest_df_copy.sort_values("AQI", ascending=False).iloc[0]
    best_row = latest_df_copy.sort_values("AQI", ascending=True).iloc[0]

    st.markdown("### AQI Gauges (Latest)")
    g1, g2, g3 = st.columns(3)
    with g1:
        aqi_gauge(city_mean_aqi, "Citywide Mean AQI")
    with g2:
        aqi_gauge(float(poor_row["AQI"]), f"poor: {poor_row['station_name']}")
    with g3:
        aqi_gauge(float(best_row["AQI"]), f"Best: {best_row['station_name']}")

elif choice == "Stations":
    st.subheader("Station Explorer")
    selected = station_selector(df)
    if selected:
        dsel = df[df["station_id"].isin(selected)].copy()
        dsel_last24 = df_last24[df_last24["station_id"].isin(selected)].copy()

        # Gauges for selected stations
        st.markdown("### AQI Gauges (Selected Stations – Latest)")
        latest_sel = df_latest[df_latest["station_id"].isin(selected)].copy()
        if not latest_sel.empty:
            _, _, overall_sel = aqi_from_pm(latest_sel["pm25"], latest_sel["pm10"])
            latest_sel["AQI"] = overall_sel.values
            ids = latest_sel["station_id"].tolist()
            for i in range(0, len(ids), 3):
                cols = st.columns(3)
                subset = latest_sel.iloc[i:i+3]
                for j, r in enumerate(subset.itertuples(index=False)):
                    with cols[j]:
                        aqi_gauge(float(r.AQI), f"{r.station_name} ({r.station_id})")

        tabs = st.tabs(["Last 24 hours", "Hourly (14d)", "Daily (14d)", "Temperature & Humidity", "Selected Pollutant (24h)"])
        with tabs[0]:
            p25_aqi, p10_aqi, overall = aqi_from_pm(dsel_last24["pm25"], dsel_last24["pm10"])
            dt = dsel_last24[["timestamp", "station_name"]].copy(); dt["AQI"] = overall.values
            fig = px.line(dt, x="timestamp", y="AQI", color="station_name", title="AQI (PM-based) – last 24 hours")
            st.plotly_chart(fig, use_container_width=True)
            line_with_threshold(dsel_last24, "pm25", "PM₂.₅ – last 24 hours", WHO_PM25_24H, "µg/m³", color_col="station_name")
            line_with_threshold(dsel_last24, "pm10", "PM₁₀ – last 24 hours", WHO_PM10_24H, "µg/m³", color_col="station_name")
        with tabs[1]:
            hourly = resample_and_classify(dsel, "H")
            if "AQI_overall" in hourly:
                fig = px.line(hourly, x="timestamp", y="AQI_overall", color="station_name", title="Hourly AQI (14 days)")
                st.plotly_chart(fig, use_container_width=True)
        with tabs[2]:
            daily = resample_and_classify(dsel, "D")
            if "AQI_overall" in daily:
                fig = px.bar(daily, x="timestamp", y="AQI_overall", color="station_name", barmode="group", title="Daily AQI (14 days)")
                st.plotly_chart(fig, use_container_width=True)
        with tabs[3]:
            fig_t = px.line(dsel_last24, x="timestamp", y="temperature", color="station_name", title="Temperature – last 24 hours (°C)")
            st.plotly_chart(fig_t, use_container_width=True)
            fig_h = px.line(dsel_last24, x="timestamp", y="humidity", color="station_name", title="Relative Humidity – last 24 hours (%)")
            st.plotly_chart(fig_h, use_container_width=True)
        with tabs[4]:
            sel_col = POLLUTANT_COLUMNS[selected_pollutant]
            if sel_col in dsel_last24:
                ttl = f"{selected_pollutant} – last 24 hours"
                line_with_threshold(
                    dsel_last24.rename(columns={sel_col: "value"}),
                    "value", ttl,
                    who_line=WHO_PM25_24H if sel_col=="pm25" else (WHO_PM10_24H if sel_col=="pm10" else None),
                    yaxis_title=POLLUTANT_UNITS[selected_pollutant],
                    color_col="station_name"
                )
            else:
                st.info(f"No data column for {selected_pollutant} found.")

elif choice == "Trends":
    st.subheader("Citywide Trends – Hourly & Daily Averages")
    hourly = resample_and_classify(df, "H")
    daily = resample_and_classify(df, "D")
    c1, c2 = st.columns(2)
    if "AQI_overall" in hourly:
        with c1:
            fig = px.line(hourly.groupby("timestamp")["AQI_overall"].mean().reset_index(), x="timestamp", y="AQI_overall", title="Citywide Hourly AQI (mean across stations)")
            st.plotly_chart(fig, use_container_width=True)
    if "AQI_overall" in daily:
        with c2:
            fig = px.bar(daily.groupby("timestamp")["AQI_overall"].mean().reset_index(), x="timestamp", y="AQI_overall", title="Citywide Daily AQI (mean across stations)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("WHO Threshold Context (Last 24 hours)")
    city24 = df_last24.groupby("timestamp")[["pm25", "pm10"]].mean().reset_index()
    line_with_threshold(city24, "pm25", "PM₂.₅ vs WHO 24h", WHO_PM25_24H, "µg/m³")
    line_with_threshold(city24, "pm10", "PM₁₀ vs WHO 24h", WHO_PM10_24H, "µg/m³")

    # New: Selected pollutant trend
    st.markdown("---")
    st.subheader(f"Selected Pollutant – {selected_pollutant} (Last 24 hours)")
    sel_col = POLLUTANT_COLUMNS[selected_pollutant]
    if sel_col in df_last24:
        city24_sel = df_last24.groupby("timestamp")[sel_col].mean().reset_index()
        line_with_threshold(
            city24_sel, sel_col,
            f"{selected_pollutant} (mean across stations)",
            who_line=WHO_PM25_24H if sel_col=="pm25" else (WHO_PM10_24H if sel_col=="pm10" else None),
            yaxis_title=POLLUTANT_UNITS[selected_pollutant]
        )

elif choice == "Data":
    st.subheader("Data Browser & Download")
    stations = df[["station_id", "station_name"]].drop_duplicates().sort_values("station_name")
    station_opt = ["All"] + stations.station_id.tolist()
    pick = st.selectbox("Station", station_opt)
    start_date = st.date_input("Start date", value=(df["timestamp"].min()).date())
    end_date = st.date_input("End date", value=(df["timestamp"].max()).date())
    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    dshow = df[mask]
    if pick != "All":
        dshow = dshow[dshow["station_id"] == pick]
    st.dataframe(dshow.sort_values(["timestamp", "station_id"]), use_container_width=True)
    csv = dshow.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="dar_air_quality_filtered.csv", mime="text/csv")

elif choice == "Reports":
    st.subheader("📑 Generate Air Quality Report")
    stations_df = df[["station_id","station_name"]].drop_duplicates().sort_values("station_name")
    station_opt = ["All"] + stations_df.station_id.tolist()

    pick = st.multiselect("Select stations", station_opt, default=["All"])
    colr1, colr2, colr3 = st.columns(3)
    with colr1: last_24 = st.button("Last 24 hours")
    with colr2: last_7d = st.button("Last 7 days")
    with colr3: last_14d = st.button("Last 14 days")

    default_start = (df["timestamp"].max() - pd.Timedelta(days=7)).date()
    default_end = df["timestamp"].max().date()
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)
    if last_24:
        start_date = (df["timestamp"].max() - pd.Timedelta(days=1)).date()
        end_date = df["timestamp"].max().date()
    if last_7d:
        start_date = (df["timestamp"].max() - pd.Timedelta(days=7)).date()
        end_date = df["timestamp"].max().date()
    if last_14d:
        start_date = (df["timestamp"].max() - pd.Timedelta(days=14)).date()
        end_date = df["timestamp"].max().date()

    st.markdown("When you click **Generate**, the PDF includes city logo, color legend, key stats, and per-station averages.")

    if st.button("Generate PDF Report"):
        mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
        dsel = df[mask]
        if "All" not in pick:
            dsel = dsel[dsel["station_id"].isin(pick)]
        if dsel.empty:
            st.warning("No data for selected range/stations.")
        else:
            pdf_bytes = generate_report(dsel, pick, pd.to_datetime(start_date), pd.to_datetime(end_date))
            st.download_button(
                "Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"dar_air_quality_report_{start_date}_{end_date}.pdf",
                mime="application/pdf"
            )

else:
    st.subheader("About – Dar es Salaam City Council Air Quality Project")
    st.image("https://i.ibb.co/gLc9tqzN/download.jpg", width=140)
    st.markdown(
        """
        **Dar es Salaam City Council (DCC)** is investing in data-driven environmental management. 
        This dashboard aggregates measurements from 15 stations across the metropolitan area, 
        including **PM₁**, **PM₂.₅**, **PM₁₀**, **NO₂**, **O₃**, **temperature**, and **humidity**.
        AQI classification follows U.S. EPA guidance, WHO 2021 24-hour guidelines shown for PM₂.₅/PM₁₀.
        
        **How to use**
        - Overview: citywide map with AQI and pollutant layers (expand/collapse).
        - Stations: compare sensors and 14-day aggregates.
        - Trends: hourly/daily city averages and WHO context.
        - Data: download raw/filtered data.
        """
    )

st.markdown(
    """
    ---
    <small>© Dar es Salaam City Council • Built with Streamlit & Folium • For inquiries, contact the DCC Environmental Unit.</small>
    """,
    unsafe_allow_html=True,
)
