#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------
# Org Navigator - Reporting Lines (Streamlit)
# v4: iPadOS-inspired warm UI + Improved Hierarchy (collision-free tidy tree)
# For: Arij Tanveer (HDS)
# ------------------------------------------------------------

import os
import io
import re
from typing import List, Dict, Tuple, Optional, Set

import pandas as pd
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================== THEME & PAGE ===============================
st.set_page_config(
    page_title="Organization Chart • Reporting Lines",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Animated warm gradient
CUSTOM_CSS = """
<style>
:root {
  --bg1: #ffefe6;
  --bg2: #ffe1d1;
  --bg3: #ffd4bd;
  --bg4: #ffdcb8;

  --glass: rgba(255,255,255,0.62);
  --stroke: rgba(0,0,0,0.06);
  --text: #1e293b;     /* slate-800 */
  --muted: #5f6b7a;    /* slate-500 */
  --accent: #ff7a37;   /* warm orange */
  --accent2: #ff9f43;  /* amber */
  --accent3: #ff6e7f;  /* coral-rose */
  --accent4: #ffc371;  /* warm gold */

  --shadow-lg: 0 28px 68px rgba(0,0,0,0.10), 0 10px 28px rgba(0,0,0,0.06);
  --shadow:   0 12px 30px rgba(0,0,0,0.08), 0 3px 12px rgba(0,0,0,0.05);
  --radius: 18px;
  --radius-sm: 14px;
  --radius-lg: 26px;
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI",
          Inter, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji",
          "Segoe UI Emoji", sans-serif;
}

@keyframes bgflow {
  0%   {background-position: 0% 50%;}
  50%  {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

html, body {
  background: linear-gradient(135deg, var(--bg1), var(--bg2), var(--bg3), var(--bg4));
  background-size: 300% 300%;
  animation: bgflow 18s ease-in-out infinite;
  font-family: var(--font);
}

.block-container { padding-top: 0.2rem; max-width: 1400px; }

header[data-testid="stHeader"] { background: transparent; }

/* Masthead (larger logo) */
.mast {
  margin: 10px 0 16px 0;
  padding: 16px 18px;
  border-radius: var(--radius-lg);
  background: linear-gradient(145deg, rgba(255,210,186,0.95), rgba(255,194,166,0.92));
  box-shadow: var(--shadow-lg);
  display: flex; align-items: center; gap: 16px; color: #3b2b1f;
}

.mast .logo {
  width: 100px; height: 100px; border-radius: 24px;
  object-fit: cover; border: 1px solid rgba(255,255,255,0.75);
  box-shadow: 0 14px 36px rgba(0,0,0,0.14);
  background: #fff;
}

.mast .title {
  font-weight: 700; font-size: 1.4rem; letter-spacing: .2px;
}

/* Glass cards */
.glass {
  background: var(--glass);
  backdrop-filter: saturate(1.25) blur(14px);
  -webkit-backdrop-filter: saturate(1.25) blur(14px);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 14px 16px;
  transition: transform .2s ease, box-shadow .2s ease;
}
.glass:hover { transform: translateY(-2px); box-shadow: 0 16px 40px rgba(0,0,0,0.10); }

.stat-card {
  background: var(--glass);
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  padding: 14px 16px;
  box-shadow: var(--shadow);
}

/* Chip / Breadcrumb */
.badge {
  display:inline-flex; align-items:center; gap:6px;
  padding:6px 12px; border-radius:999px; font-size:0.78rem;
  color:#6b3b1f; background:#fff2e8; border:1px solid #ffd9c2; margin-right:6px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
  background: #1f2937; color: #fff; border: 0; padding: 10px 16px;
  border-radius: 12px; box-shadow: var(--shadow); font-weight: 600;
}
.stButton>button:hover, .stDownloadButton>button:hover { background: #111827; }

/* iPad-like Selects (widget look) */
div[data-baseweb="select"] > div {
  border-radius: 14px !important; border: 1px solid #f2d3c3 !important;
  background: rgba(255,255,255,0.9);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.6), 0 4px 10px rgba(0,0,0,0.06);
}
div[data-baseweb="select"] > div:focus-within { box-shadow: 0 0 0 3px rgba(251,146,60,0.35); }

.stTextInput>div>div>input {
  border-radius: 14px !important; border: 1px solid #f2d3c3 !important;
  background: rgba(255,255,255,0.95);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.6), 0 4px 10px rgba(0,0,0,0.05);
}
.stTextInput>div>div>input:focus { box-shadow: 0 0 0 3px rgba(251,146,60,0.35); }

/* Segmented control (views) */
.segmented .stRadio > div {
  display: inline-flex; padding: 4px; border-radius: 999px;
  background: rgba(255,255,255,0.55); border: 1px solid #f1cdb7; box-shadow: var(--shadow);
}
.segmented .stRadio > div > label {
  margin: 0 2px !important;
}
.segmented .stRadio > div > label > div:nth-child(2) {
  padding: 8px 14px; border-radius: 999px;
}
.segmented .stRadio > div > label[data-checked="true"] > div:nth-child(2) {
  background: linear-gradient(135deg, var(--accent3), var(--accent4));
  color: #fff !important; font-weight: 700;
}

/* Slider track */
.stSlider>div>div>div>div { background: linear-gradient(90deg, #ff7a37, #ff9f43) !important; }
.stSlider>div>div>div>div>div { background: #fff; border:1px solid #ffd9c2; }

/* iOS-like toggles */
.stToggle {
  --toggle-width: 56px; --toggle-height: 30px;
}
.stToggle label div[role="switch"] {
  width: var(--toggle-width) !important; height: var(--toggle-height) !important;
  background: #ffe1cf !important; border-radius: 999px !important;
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.08);
  transition: background .2s ease;
}
.stToggle label div[role="switch"][aria-checked="true"] { background: #ff7a37 !important; }
.stToggle label div[role="switch"]::after {
  content: ""; width: 26px; height: 26px; background: #fff; border-radius: 999px; display:block; margin: 2px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.18);
  transition: transform .2s ease;
}
.stToggle label div[role="switch"][aria-checked="true"]::after { transform: translateX(26px); }

footer, #MainMenu {visibility:hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================== DATA LOADING ===============================

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file_or_path: Optional[io.BytesIO]) -> pd.DataFrame:
    """Load and normalize CSV into a consistent schema. Handles duplicate manager ID headers."""
    if uploaded_file_or_path is None:
        default_path = "Reporting Lines.csv"
        if os.path.exists(default_path):
            df = pd.read_csv(default_path, dtype=str, encoding_errors="ignore")
        else:
            raise FileNotFoundError("Upload CSV or place 'Reporting Lines.csv' with Reporting.py.")
    else:
        df = pd.read_csv(uploaded_file_or_path, dtype=str, encoding_errors="ignore")

    cols = list(df.columns)

    def norm(c: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (c or "").strip().lower())

    def find_all(label: str) -> List[str]:
        out = []
        for c in cols:
            if c.strip().lower() == label.strip().lower() or norm(c) == norm(label):
                out.append(c)
        return out

    def pick(*cands) -> Optional[str]:
        for c in cands:
            m = find_all(c)
            if m: return m[0]
        # contains fallback
        for c in cols:
            for cand in cands:
                if norm(cand) in norm(c): return c
        return None

    emp_id_col  = pick("Employee Number", "Employee No", "Emp No")
    name_col    = pick("Employee Name", "Name")
    org_col     = pick("Org Unit", "Organization Unit", "OrgUnit")
    area_col    = pick("Personnel Area", "Area")
    sub_col     = pick("Personnel SubArea", "SubArea", "Sub Area")

    mgr_nm_admin_col = pick("Name ( Admin Reporting)")
    mgr_nm_line_col  = pick("Name ( Manager)")

    mgr_id_candidates = find_all("Employee No (Manager)")
    # Heuristic: first occurrence -> Admin ID, second -> Line ID (if exists)
    mgr_id_admin_col = mgr_id_candidates[0] if len(mgr_id_candidates) >= 1 else None
    mgr_id_line_col  = mgr_id_candidates[1] if len(mgr_id_candidates) >= 2 else (mgr_id_candidates[0] if mgr_id_candidates else None)

    nd = pd.DataFrame()
    nd["emp_id"] = df[emp_id_col].astype(str).str.strip() if emp_id_col else ""
    nd["name"] = df[name_col].astype(str).str.strip() if name_col else "Unknown"
    nd["org_unit"] = (df[org_col].astype(str).str.strip() if org_col else "")
    nd["area"] = (df[area_col].astype(str).str.strip() if area_col else "")
    nd["sub_area"] = (df[sub_col].astype(str).str.strip() if sub_col else "")

    nd["mgr_admin_id"] = df[mgr_id_admin_col].astype(str).str.strip() if mgr_id_admin_col else ""
    nd["mgr_admin_name"] = df[mgr_nm_admin_col].astype(str).str.strip() if mgr_nm_admin_col else ""
    nd["mgr_id"] = df[mgr_id_line_col].astype(str).str.strip() if mgr_id_line_col else ""
    nd["mgr_name"] = df[mgr_nm_line_col].astype(str).str.strip() if mgr_nm_line_col else ""

    def clean_id(x: str) -> str:
        x = (x or "").strip()
        return "" if x in ("", "0", "nan", "NaN", "None", "NONE") else x

    for c in ["emp_id", "mgr_admin_id", "mgr_id"]:
        nd[c] = nd[c].map(clean_id)

    nd = nd.dropna(subset=["emp_id"]).query("emp_id != ''").copy()
    nd = nd.sort_values("emp_id").drop_duplicates("emp_id", keep="first")
    nd["name"] = nd["name"].fillna("")
    nd.loc[nd["name"].eq("") , "name"] = "Unknown"

    for c in ["org_unit","area","sub_area","mgr_admin_name","mgr_name"]:
        nd[c] = nd[c].fillna("").astype(str)

    return nd

# =============================== GRAPH UTILS ===============================

def build_graph(df: pd.DataFrame, mode: str) -> nx.DiGraph:
    mgr_id_col = "mgr_id" if mode == "Manager (Line)" else "mgr_admin_id"
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_node(r["emp_id"], label=r["name"], org_unit=r["org_unit"], area=r["area"], sub_area=r["sub_area"])
    ids = set(df["emp_id"])
    for _, r in df.iterrows():
        m = r[mgr_id_col]
        if m and m in ids:
            G.add_edge(m, r["emp_id"])  # manager -> employee
    return G

def find_roots(G: nx.DiGraph) -> List[str]:
    return [n for n in G.nodes if G.in_degree(n) == 0]

def get_subtree_nodes(G: nx.DiGraph, root: str, depth: int = 3) -> Set[str]:
    nodes = {root}
    frontier = {root}
    for _ in range(depth):
        nxt = set()
        for u in frontier:
            nxt.update(G.successors(u))
        nodes.update(nxt)
        frontier = nxt
    return nodes

def compute_descendant_counts(G: nx.DiGraph) -> Dict[str, int]:
    try:
        return {n: len(nx.descendants(G, n)) for n in G.nodes}
    except Exception:
        return {n: 0 for n in G.nodes}

def filtered_df(df: pd.DataFrame, orgs: List[str], areas: List[str], subs: List[str]) -> pd.DataFrame:
    out = df
    if orgs:  out = out[out["org_unit"].isin(orgs)]
    if areas: out = out[out["area"].isin(areas)]
    if subs:  out = out[out["sub_area"].isin(subs)]
    return out

# =============================== EXISTING VIS HELPERS ===============================

def to_pyvis_html(G: nx.DiGraph, sub_nodes: Set[str], height_px: int = 700) -> str:
    H = G.subgraph(sub_nodes).copy()
    net = Network(height=f"{height_px}px", width="100%", directed=True, notebook=False, bgcolor="#ffffff", font_color="#2f3e5b")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=110, spring_strength=0.02, damping=0.8, overlap=0.1)

    warm_palette = ["#ff6e7f","#ff7a37","#ff9f43","#f59e0b","#fbbf24","#fda4af","#fcd34d","#fca38a","#f59f7a","#f8b4a6"]
    orgs = list({H.nodes[n].get("org_unit","") for n in H.nodes})
    color_map = {ou: warm_palette[i % len(warm_palette)] for i, ou in enumerate(sorted(orgs))}

    for n, attrs in H.nodes(data=True):
        label = attrs.get("label", n)
        org = attrs.get("org_unit", "")
        area = attrs.get("area", "")
        sub = attrs.get("sub_area", "")
        title = f"<b>{label}</b><br>ID: {n}<br>Org: {org}<br>Area: {area}<br>Sub-Area: {sub}"
        net.add_node(n, label=label, title=title, color=color_map.get(org, "#e5e7eb"), shape="dot", size=18)
    for u, v in H.edges():
        net.add_edge(u, v, arrows="to", color="#b3bfd2")
    return net.generate_html(notebook=False)

def build_plot_df(G: nx.DiGraph, root: str, sub_nodes: Set[str]) -> pd.DataFrame:
    rows = [{"id":"ROOT","parent":"","label":"Organization","org_unit":""}]
    for n in sub_nodes:
        preds = list(G.predecessors(n))
        parent = "ROOT" if (not preds or n == root) else preds[0]
        a = G.nodes[n]
        rows.append({
            "id": n,
            "parent": parent,
            "label": a.get("label", n),
            "org_unit": a.get("org_unit",""),
            "area": a.get("area",""),
            "sub_area": a.get("sub_area",""),
        })
    return pd.DataFrame(rows)

# =============================== HIERARCHY v4 (collision-free tidy) ===============================

def _children_sorted(G: nx.DiGraph, sub: Set[str]) -> Dict[str, List[str]]:
    return {n: sorted([c for c in G.successors(n) if c in sub],
                      key=lambda c: (G.nodes[c].get("label","").lower(), c)) for n in sub}

def _visible_text(name: str, mode: str) -> str:
    if mode == "Hover only":
        return ""
    if mode == "Initials":
        parts = [p for p in re.split(r"\\s+", name.strip()) if p]
        if not parts: return ""
        if len(parts) == 1: return parts[0][:2].upper()
        return (parts[0][0] + parts[-1][0]).upper()
    return name  # Full

def _wrap_label(txt: str, width: int = 18) -> str:
    """Wrap text with <br> to avoid ultra-wide labels."""
    if not txt: return ""
    out, line = [], ""
    for w in txt.split():
        if len(line) + len(w) + 1 <= width:
            line = (line + " " + w).strip()
        else:
            out.append(line); line = w
    if line: out.append(line)
    return "<br>".join(out)

def _text_unit_width_for_label(label: str, mode: str, wrap_width: int) -> float:
    """
    Estimate width units for a label after wrapping (used for collision avoidance).
    We approximate width by longest wrapped line length.
    """
    if mode == "Hover only":
        return 1.5
    if mode == "Initials":
        return 1.8
    # Full
    # compute longest line after wrapping
    words = label.split()
    lines, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 <= wrap_width:
            line = (line + " " + w).strip()
        else:
            lines.append(line); line = w
    if line: lines.append(line)
    max_len = max((len(s) for s in lines), default=2)
    # char-to-unit factor ~ 0.10 + padding
    return min(7.0, max(2.0, 0.10 * max_len + 1.2))

def _layout_tidy_width_aware(
    G: nx.DiGraph,
    root: str,
    sub_nodes: Set[str],
    label_mode: str,
    wrap_width: int,
    h_gap_units: float = 0.5
) -> Tuple[Dict[str, Tuple[float,float]], Dict[str,int], Dict[str,float]]:
    """
    Compute tidy layout with width-aware spacing:
    - Leaves placed sequentially with their width and gaps
    - Parents centered above children
    Returns: pos_units (x,y in units), level map, label_width_units
    """
    children = _children_sorted(G, sub_nodes)
    level: Dict[str,int] = {root: 0}
    q = [root]
    while q:
        u = q.pop(0)
        for v in children.get(u, []):
            if v not in level:
                level[v] = level[u] + 1
                q.append(v)

    label_width: Dict[str, float] = {}
    for n in sub_nodes:
        nm = G.nodes[n].get("label", n)
        label_width[n] = _text_unit_width_for_label(nm, label_mode, wrap_width)

    x: Dict[str, float] = {}
    cursor = 0.0

    def place(n: str) -> float:
        nonlocal cursor
        kids = children.get(n, [])
        if not kids:
            w = label_width[n]
            x[n] = cursor + w/2.0
            cursor += w + h_gap_units
            return w
        total = 0.0
        child_centers = []
        for k in kids:
            cw = place(k)
            total += cw
            child_centers.append(x[k])
        total += h_gap_units * max(0, len(kids)-1)
        x[n] = sum(child_centers)/len(child_centers)
        total = max(total, label_width[n])  # ensure parent not thinner than span
        return total

    place(root)

    xmin = min(x.values())
    x = {n: v - xmin for n, v in x.items()}
    pos_units = {n: (x[n], level.get(n,0)) for n in sub_nodes}
    return pos_units, level, label_width

def _enforce_per_level_spacing(
    pos_units: Dict[str, Tuple[float,float]],
    level: Dict[str,int],
    label_width_units: Dict[str, float],
    min_margin: float = 0.4
) -> Dict[str, Tuple[float,float]]:
    """
    Prevent label overlap by scaling each level horizontally about its mean
    so that neighboring centers are at least (w_i/2 + w_j/2 + min_margin) apart.
    This keeps parent-child vertical alignment acceptable while eliminating collisions.
    """
    # group by level
    by_lv: Dict[int, List[str]] = {}
    for n, (_, y) in pos_units.items():
        lv = level.get(n, int(y))
        by_lv.setdefault(lv, []).append(n)

    new_x: Dict[str, float] = {n: pos_units[n][0] for n in pos_units}
    for lv, nodes in by_lv.items():
        nodes.sort(key=lambda n: new_x[n])
        if len(nodes) < 2:
            continue
        # compute minimum required gaps vs actual gaps; determine a scale factor
        xs = [new_x[n] for n in nodes]
        req = []
        for i in range(len(nodes)-1):
            left, right = nodes[i], nodes[i+1]
            need = (label_width_units[left]/2.0 + label_width_units[right]/2.0 + min_margin)
            have = xs[i+1] - xs[i]
            if have < need:
                req.append(need / max(have, 1e-6))
            else:
                req.append(1.0)
        s = max(req) if req else 1.0
        if s > 1.0001:
            # scale about mean to preserve center placement
            mean_x = sum(xs)/len(xs)
            for n in nodes:
                new_x[n] = (new_x[n] - mean_x)*s + mean_x

    # write back positions
    out = {n: (new_x[n], pos_units[n][1]) for n in pos_units}
    return out

def plot_hierarchy_tidy(
    G: nx.DiGraph,
    root: str,
    sub_nodes: Set[str],
    node_size: int = 18,
    h_sep: float = 1.8,
    v_sep: float = 1.25,
    label_mode: str = "Full",  # "Full" | "Initials" | "Hover only"
    wrap_width: int = 16,
    max_per_level: int = 240,
    orthogonal_edges: bool = True,
    height: int = 760
) -> go.Figure:
    """
    Top-down tidy hierarchy with:
    - width-aware layout
    - per-level horizontal scaling to ensure no label collisions
    - optional orthogonal (elbow) edges
    """
    pos_units, level, label_width_units = _layout_tidy_width_aware(
        G, root, sub_nodes, label_mode, wrap_width, h_gap_units=0.55
    )

    # Enforce spacing so labels can’t collide on the same level
    pos_units = _enforce_per_level_spacing(pos_units, level, label_width_units, min_margin=0.45)

    # Scale to pixels
    pos = {n: (pos_units[n][0]*h_sep, pos_units[n][1]*v_sep) for n in sub_nodes}

    # Cap crowded levels (+N more)
    by_lv: Dict[int, List[str]] = {}
    for n in sub_nodes:
        lv = level.get(n, 0)
        by_lv.setdefault(lv, []).append(n)
    for lv in by_lv:
        by_lv[lv].sort(key=lambda n: (pos[n][0], G.nodes[n].get("label","").lower(), n))

    visible: Set[str] = set()
    overflow: Dict[int, Tuple[float, int]] = {}
    for lv, nodes in by_lv.items():
        if len(nodes) <= max_per_level:
            visible.update(nodes)
        else:
            keep = max_per_level
            left = nodes[: keep // 2]
            right = nodes[-(keep - len(left)):]
            keep_nodes = left + right
            visible.update(keep_nodes)
            xs = [pos[n][0] for n in keep_nodes]
            xc = (min(xs) + max(xs)) / 2.0 if xs else 0.0
            overflow[lv] = (xc, len(nodes) - len(keep_nodes))

    # Colors
    warm_palette = ["#ff6e7f","#ff7a37","#ff9f43","#f59e0b","#fbbf24","#fda4af","#fcd34d","#fca38a","#fb7185"]
    orgs = sorted({G.nodes[n].get("org_unit","") for n in sub_nodes})
    color_map = {ou: warm_palette[i % len(warm_palette)] for i, ou in enumerate(orgs)}

    # ---------- Edges ----------
    points = []
    for u, v in G.edges():
        if u in sub_nodes and v in sub_nodes and v in visible:
            x0,y0 = pos[u]; x1,y1 = pos[v]
            if orthogonal_edges:
                ym = (y0 + y1)/2.0
                points += [(x0,y0),(x0,ym),(x1,ym),(x1,y1),(None,None)]
            else:
                points += [(x0,y0),(x1,y1),(None,None)]
    edge_x = [p[0] for p in points]
    edge_y = [p[1] for p in points]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(80,90,110,0.38)", width=1.7),
        hoverinfo="skip", showlegend=False
    )

    # ---------- Nodes & Labels ----------
    node_x, node_y, node_color, node_text, node_hover = [], [], [], [], []
    for n in sorted(list(visible), key=lambda z: (pos[z][1], pos[z][0])):
        a = G.nodes[n]; x,y = pos[n]
        node_x.append(x); node_y.append(y)
        org = a.get("org_unit",""); area = a.get("area",""); suba = a.get("sub_area","")
        node_color.append(color_map.get(org, "#e5e7eb"))
        vis_label = _visible_text(a.get("label", n), label_mode)
        node_text.append(_wrap_label(vis_label, wrap_width) if label_mode!="Hover only" else "")
        node_hover.append(f"<b>{a.get('label', n)}</b><br>ID: {n}<br>Org: {org}<br>Area: {area}<br>Sub-Area: {suba}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text' if label_mode!="Hover only" else 'markers',
        marker=dict(size=node_size, color=node_color, line=dict(color="#ffffff", width=1.1)),
        text=node_text if label_mode!="Hover only" else None,
        textposition="top center", textfont=dict(size=12, color="#1f2937"),
        hovertext=node_hover, hoverinfo="text", showlegend=False
    )

    # Overflow badges
    over_x, over_y, over_txt = [], [], []
    for lv, (xc,cnt) in overflow.items():
        over_x.append(xc); over_y.append(lv*v_sep); over_txt.append(f"+{cnt} more")
    overflow_trace = go.Scatter(
        x=over_x, y=over_y, mode="text",
        text=[f"<b>{t}</b>" for t in over_txt],
        textposition="bottom center", textfont=dict(color="#8a4b2d", size=12),
        hoverinfo="skip", showlegend=False
    )

    fig = go.Figure(data=[edge_trace, node_trace, overflow_trace])
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=24, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white", paper_bgcolor="white",
        dragmode="pan", transition={'duration': 200}
    )
    fig.update_yaxes(autorange="reversed")

    # Auto-fit X
    xs = [x for x in node_x if x is not None]
    if xs:
        xmin, xmax = min(xs), max(xs)
        span = max(2.0, xmax - xmin + 2.2)
        fig.update_xaxes(range=[xmin - 1.1, xmin - 1.1 + span])

    return fig

# =============================== SEARCH, BREADCRUMBS, QC ===============================

def search_people(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q:
        return df.head(0)
    m = df["emp_id"].str.lower().str.contains(q, na=False) | df["name"].str.lower().str.contains(q, na=False)
    return df[m].copy()

def breadcrumbs(G: nx.DiGraph, node: str) -> List[Tuple[str, str]]:
    preds = list(G.predecessors(node))
    if not preds:
        return [(node, G.nodes[node].get("label", node))]
    p = preds[0]
    return breadcrumbs(G, p) + [(node, G.nodes[node].get("label", node))]

def quality_checks(df: pd.DataFrame, G: nx.DiGraph, mode: str) -> Dict[str, pd.DataFrame]:
    mgr_col = "mgr_id" if mode == "Manager (Line)" else "mgr_admin_id"
    known = set(df["emp_id"])

    orphans = df[(df[mgr_col].ne("")) & (~df[mgr_col].isin(known))][
        ["emp_id","name","org_unit","area","sub_area",mgr_col]
    ].rename(columns={mgr_col:"manager_id_not_found"})

    dups = df[df.duplicated("emp_id", keep=False)].sort_values("emp_id")

    cycles_list = []
    try:
        for cyc in nx.simple_cycles(G):
            if len(cyc) > 1:
                cycles_list.append(cyc)
    except Exception:
        pass
    cycles_df = pd.DataFrame({"cycle_nodes": cycles_list})

    multi_mgr = df[(df["mgr_id"].ne("")) & (df["mgr_admin_id"].ne("")) & (df["mgr_id"] != df["mgr_admin_id"])][
        ["emp_id","name","mgr_id","mgr_admin_id"]
    ]

    return {
        "Orphans (Manager not in file)": orphans,
        "Duplicate Employee Numbers": dups,
        "Cycles (if any)": cycles_df,
        "Different Admin vs Line Manager": multi_mgr
    }

# =============================== LOGO HANDLING ===============================

def find_logo_path() -> Optional[str]:
    candidates = [
        "UploadedImage4.jpg", "UploadedImage2.jpg",
        "logo.png","logo.jpg","Logo.png","Logo.jpg","brand.png","brand.jpg"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def get_logo_bytes(uploaded_logo) -> Optional[bytes]:
    if uploaded_logo is not None:
        return uploaded_logo.getvalue()
    p = find_logo_path()
    if p:
        return open(p, "rb").read()
    return None

# =============================== SIDEBAR & MAIN ===============================

st.sidebar.header("👥 Org Navigator")

uploaded_logo = st.sidebar.file_uploader("Brand logo (PNG/JPG)", type=["png","jpg","jpeg"])
file = st.sidebar.file_uploader("Upload Reporting Lines CSV", type=["csv"], accept_multiple_files=False)

try:
    df_all = load_csv(file)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

relation_choice = st.sidebar.radio(
    "Select relationship",
    ["Manager (Line)", "Admin Reporting"],
    help="Switch between line manager and administrative reporting."
)

with st.sidebar.expander("Filters", expanded=True):
    orgs  = sorted(df_all["org_unit"].dropna().unique().tolist())
    areas = sorted(df_all["area"].dropna().unique().tolist())
    subs  = sorted(df_all["sub_area"].dropna().unique().tolist())

    sel_orgs  = st.multiselect("Org Unit", orgs)
    sel_areas = st.multiselect("Personnel Area", areas)
    sel_subs  = st.multiselect("Sub-Area", subs)

df = filtered_df(df_all, sel_orgs, sel_areas, sel_subs)

G = build_graph(df, relation_choice)
roots = find_roots(G)
desc_counts = compute_descendant_counts(G)

# =============================== HEADER (BIGGER LOGO) ===============================

logo_bytes = get_logo_bytes(uploaded_logo)
col_logo, col_title = st.columns([0.12, 0.88])
with col_logo:
    st.markdown("<div class='mast'>", unsafe_allow_html=True)
    if logo_bytes:
        st.image(logo_bytes, caption=None, width=100, output_format="PNG")
    else:
        st.markdown("<div class='logo' style='width:100px;height:100px;border-radius:24px;background:#fff;border:1px solid rgba(255,255,255,0.75)'></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_title:
    # Removed previous subtitle line as requested
    st.markdown(
        """
        <div class='mast'>
          <div>
            <div class='title'>Organization Chart • Reporting Lines</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================== TOP CONTROLS (Search / Focus / View) ===============================

col_l, col_m, col_r = st.columns([2, 2, 2])

with col_l:
    q = st.text_input("🔎 Search by Name or Employee ID", placeholder="e.g., Hassan, 117, Saqib, 665 ...")
    results = search_people(df, q)
    if not results.empty:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.write(f"Matches: {len(results)}")
        st.dataframe(results[["emp_id","name","org_unit","area","sub_area"]].reset_index(drop=True), use_container_width=True, height=220)
        st.markdown("</div>", unsafe_allow_html=True)

with col_m:
    if not G.nodes:
        st.warning("No people in graph with current filters.")
        st.stop()
    default_root = max(roots, key=lambda r: desc_counts.get(r, 0)) if roots else next(iter(G.nodes))
    people_options = [(f"{G.nodes[n]['label']}  —  {n}", n) for n in sorted(G.nodes, key=lambda x: G.nodes[x].get('label',''))]
    root_choice = st.selectbox(
        "📌 Focus person (root)",
        options=[v for _, v in people_options],
        format_func=lambda v: next((t for t, val in people_options if val == v), v),
        index=next((i for i, (_, val) in enumerate(people_options) if val == default_root), 0)
    )

with col_r:
    depth = st.slider("Depth from selected root", 1, 10, 3)

# Segmented control for views (iPad-like)
st.markdown("<div class='segmented'>", unsafe_allow_html=True)
view = st.radio("View", ["Network", "Treemap", "Icicle", "Hierarchy"], horizontal=True, label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if root_choice not in G.nodes:
    st.warning("Selected person not in current graph; adjust filters.")
    st.stop()

# =============================== BREADCRUMBS & STATS ===============================

bc = breadcrumbs(G, root_choice)
bc_html = " / ".join([f"<span class='badge'>{G.nodes[i]['label']}</span>" for i, _ in bc])
st.markdown(f"{bc_html}", unsafe_allow_html=True)

direct_count = G.out_degree(root_choice)
tot_desc = desc_counts.get(root_choice, 0)
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(f"<div class='stat-card'><b>Selected</b><br>{G.nodes[root_choice].get('label','Unknown')}<br><span class='small'>ID: {root_choice}</span></div>", unsafe_allow_html=True)
with col_b:
    st.markdown(f"<div class='stat-card'><b>Direct Reports</b><br>{direct_count}</div>", unsafe_allow_html=True)
with col_c:
    st.markdown(f"<div class='stat-card'><b>Total in Subtree</b><br>{tot_desc}</div>", unsafe_allow_html=True)
with col_d:
    org_tag = G.nodes[root_choice].get("org_unit","")
    st.markdown(f"<div class='stat-card'><b>Org Unit</b><br>{org_tag or '—'}</div>", unsafe_allow_html=True)

# =============================== MAIN LAYOUT ===============================

left, right = st.columns([1.32, 1])
sub_nodes = get_subtree_nodes(G, root_choice, depth)

with left:
    if view == "Network":
        html = to_pyvis_html(G, sub_nodes, height_px=760)
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.components.v1.html(html, height=780, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)

    elif view in ("Treemap", "Icicle"):
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        plot_df = build_plot_df(G, root_choice, sub_nodes)
        if view == "Treemap":
            fig = px.treemap(plot_df, ids="id", names="label", parents="parent", color="org_unit",
                             color_discrete_sequence=px.colors.qualitative.Set3)
        else:
            fig = px.icicle(plot_df, ids="id", names="label", parents="parent", color="org_unit",
                            color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(margin=dict(l=0,r=0,t=24,b=0), height=760, transition={'duration': 200})
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        st.markdown("</div>", unsafe_allow_html=True)

    else:  # Hierarchy (Top-Down, collision-free tidy)
        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        # iPad‑style toggles & widget controls
        t1, t2, t3 = st.columns([1,1,1])
        with t1:
            label_mode = st.selectbox("Labels", ["Full", "Initials", "Hover only"], index=0,
                                      help="Use initials or hover-only to declutter very large orgs.")
        with t2:
            orthogonal = st.toggle("Orthogonal Edges", value=True, help="Elbow connectors like classic org charts.")
        with t3:
            show_ids = st.toggle("Show IDs in Labels", value=False, help="Append employee IDs to labels.")

        # Spacing & size (iOS widget vibe)
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            node_size = st.slider("Node Size", 12, 30, 18)
        with c2:
            h_sep = st.slider("Horizontal Spacing", 1.2, 4.0, 2.0, 0.1)
        with c3:
            v_sep = st.slider("Vertical Spacing", 0.9, 2.2, 1.25, 0.05)
        with c4:
            wrap_width = st.slider("Wrap (chars/line)", 10, 28, 16, 1)

        c5, c6 = st.columns([1,1])
        with c5:
            max_per_level = st.slider("Max nodes per level", 20, 600, 260, 10,
                                      help="Caps crowded levels and shows '+N more'.")
        with c6:
            chart_h = st.slider("Chart Height (px)", 520, 1600, 780, 10)

        # Temporary label augmentation if requested
        if label_mode != "Hover only" and show_ids:
            for n in sub_nodes:
                a = G.nodes[n]
                a["__orig"] = a.get("label","")
                a["label"] = f"{a.get('__orig','')}  ({n})"

        fig = plot_hierarchy_tidy(
            G, root_choice, sub_nodes,
            node_size=node_size, h_sep=h_sep, v_sep=v_sep,
            label_mode=label_mode, wrap_width=wrap_width,
            max_per_level=max_per_level, orthogonal_edges=orthogonal, height=chart_h
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        if label_mode != "Hover only" and show_ids:
            for n in sub_nodes:
                a = G.nodes[n]
                if "__orig" in a:
                    a["label"] = a["__orig"]
                    del a["__orig"]

        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.subheader("👤 Team")
    def node_row(c):
        a = G.nodes[c]
        return (c, a.get("label",""), a.get("org_unit",""), a.get("area",""), a.get("sub_area",""),
                G.out_degree(c), desc_counts.get(c,0))
    direct_list = list(G.successors(root_choice))
    direct_df = pd.DataFrame([node_row(c) for c in direct_list], columns=["emp_id","name","org_unit","area","sub_area","direct_reports","total_descendants"])
    other_nodes = [n for n in sorted(list(sub_nodes)) if n not in set(direct_list+[root_choice])]
    others_df = pd.DataFrame([node_row(c) for c in other_nodes], columns=["emp_id","name","org_unit","area","sub_area","direct_reports","total_descendants"])

    with st.expander(f"Direct Reports ({len(direct_df)})", expanded=True):
        st.dataframe(direct_df.reset_index(drop=True), use_container_width=True, height=260)
    with st.expander(f"Others in Subtree ({len(others_df)})", expanded=False):
        st.dataframe(others_df.reset_index(drop=True), use_container_width=True, height=320)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    if st.button("⬇️ Export this subtree (CSV)"):
        subtree_emp_ids = list(sub_nodes)
        export_df = df[df["emp_id"].isin(subtree_emp_ids)].copy()
        buf = io.BytesIO()
        export_df.to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label="Download CSV",
            data=buf.getvalue(),
            file_name=f"org_subtree_{root_choice}.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# =============================== QUALITY CHECKS ===============================

st.markdown("---")
st.subheader("🧪 Data Quality Checks")
qc = quality_checks(df, G, relation_choice)
qcols = st.columns(4)
keys = list(qc.keys())
for i, k in enumerate(keys):
    with qcols[i % 4]:
        st.markdown(f"**{k}**")
        small = qc[k]
        if small.empty:
            st.success("None 🎉")
        else:
            st.dataframe(small.reset_index(drop=True), use_container_width=True, height=220)

st.caption("Tip: If labels ever feel tight, increase Horizontal Spacing and reduce Wrap width; or try Initials / Hover-only.")
