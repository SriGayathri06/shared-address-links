from pathlib import Path
import streamlit as st
import pandas as pd
from pyvis.network import Network
from streamlit.components.v1 import html

# ---------- Page config ----------
st.set_page_config(
    page_title="Shared Address Links",
    layout="wide",
    initial_sidebar_state="collapsed",
)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_NODES = DATA_DIR / "nodes.csv"
DEFAULT_EDGES = DATA_DIR / "edges.csv"

# ---------- Helpers ----------
@st.cache_data
def load_data(nodes_path: str, edges_path: str):
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    return nodes, edges

def ensure_artifacts():
    """Run prep once if nodes/edges don't exist (optional)."""
    if not (DEFAULT_NODES.exists() and DEFAULT_EDGES.exists()):
        import subprocess, sys
        prep_path = BASE_DIR / "prep.py"
        if prep_path.exists():
            # run prep.py in-process env
            result = subprocess.run([sys.executable, str(prep_path)], capture_output=True, text=True)
            st.toast("Generated nodes/edges from prep.py")
            if result.returncode != 0:
                st.error(result.stderr)
        else:
            st.warning("prep.py not found; please upload nodes/edges manually.")

# Allow the iframe to size by our slider, but not exceed container width
st.markdown(
    """
    <style>
      .stApp iframe { max-width: 100% !important; }
      .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Sidebar (collapsible) ----------
with st.sidebar:
    st.title("Controls")

    with st.expander("Data", expanded=True):
        use_defaults = st.toggle("Use bundled data (recommended)", value=True)
        if use_defaults:
            nodes_path = str(DEFAULT_NODES)
            edges_path = str(DEFAULT_EDGES)
            ensure_artifacts()  # <-- ensure outputs exist
        else:
            nodes_path = st.text_input("Nodes CSV", str(DEFAULT_NODES))
            edges_path = st.text_input("Edges CSV", str(DEFAULT_EDGES))

    nodes, edges = load_data(nodes_path, edges_path)


    with st.expander("Filters", expanded=True):
        # contributor type filter
        types = sorted([t for t in nodes["contrib_type"].dropna().unique()])
        default_types = [t for t in types if str(t).lower() == "individual"] or types
        sel_types = st.multiselect("Contributor Type(s)", types, default=default_types)

        # min contributors per address
        min_contribs = st.slider("Min contributors at an address", 2, 15, 2, 1)

        # amount range (for nodes)
        amt_min = float(nodes["total_amount"].min())
        amt_max = float(nodes["total_amount"].max())
        sel_amt = st.slider("Node total amount range", amt_min, amt_max, (amt_min, amt_max))

        # graph size controls
        graph_h = st.slider("Graph height (px)", 600, 1400, 900, 50)
        graph_w = st.slider("Graph width (px)", 600, 1800, 1200, 50)

# ---------- Filtering ----------
addr_contribs = edges.groupby("target").agg(contributors=("source", "nunique")).reset_index()
addr_keep = set(addr_contribs.loc[addr_contribs["contributors"] >= min_contribs, "target"])

is_addr = nodes["type"] == "address"
is_person = nodes["type"] == "contributor"

persons_ok = is_person
if sel_types:
    persons_ok = is_person & nodes["contrib_type"].isin(sel_types)

nodes_f = nodes[(is_addr & nodes["id"].isin(addr_keep)) | persons_ok].copy()
nodes_f = nodes_f[nodes_f["total_amount"].between(sel_amt[0], sel_amt[1])]

keep_ids = set(nodes_f["id"])
edges_f = edges[edges["source"].isin(keep_ids) & edges["target"].isin(keep_ids)].copy()

# ---------- Header ----------
st.title("Shared Address Links — Interactive")

# ---------- Summary ----------
colA, colGap = st.columns([2, 1])
with colA:
    st.subheader("Summary")
    n_addr = int((nodes_f["type"] == "address").sum())
    n_people = int((nodes_f["type"] == "contributor").sum())
    st.metric("Addresses shown", n_addr)
    st.metric("Contributors shown", n_people)

    st.markdown("**Top Shared Addresses** (by #contributors):")
    top = (
        edges_f.groupby(["target"])
        .agg(contributors=("source", "nunique"), tx=("source", "size"))
        .reset_index()
        .merge(
            nodes_f[nodes_f["type"] == "address"][["id", "label"]],
            left_on="target",
            right_on="id",
            how="left",
        )
        .sort_values(["contributors", "tx"], ascending=False)
        .head(12)[["label", "contributors", "tx"]]
    )
    st.dataframe(top, hide_index=True, use_container_width=True)

# ---------- Network (full width, slider-controlled size) ----------
st.subheader("Interactive Network")
net = Network(height=f"{graph_h}px", width=f"{graph_w}px", bgcolor="#ffffff", font_color="#000000")
net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150,
               spring_strength=0.01, damping=0.09)

# nodes
for _, r in nodes_f.iterrows():
    title = f"{'Address' if r['type']=='address' else r.get('contrib_type','Contributor')}"
    title += f" • {int(r.get('tx_count',0))} tx • ${float(r.get('total_amount',0)):,.0f}"
    if r["type"] == "address":
        net.add_node(r["id"], label=r["label"], title=title, shape="square")
    else:
        net.add_node(r["id"], label=r["label"], title=title)

# edges
for _, e in edges_f.iterrows():
    etitle = f"{e['address']} • {int(e['tx_count'])} tx • ${float(e['total_amount']):,.0f}"
    net.add_edge(e["source"], e["target"], title=etitle, value=float(e["tx_count"]))

net.save_graph("network.html")
with open("network.html", "r", encoding="utf-8") as f:
    html(f.read(), height=graph_h + 40, width=graph_w, scrolling=True)
