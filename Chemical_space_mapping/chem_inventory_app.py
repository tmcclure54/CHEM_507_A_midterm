
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem import SDWriter
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.svm import LinearSVC
from sklearn.manifold import Isomap
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(page_title="Chemical Library Selector", layout="wide")

# ===================== Header with About popover =====================
def about_popover():
    content = r"""
### About this app
**Goal:** help chemists upload a scope and quickly **select the most diverse N** compounds (coverage) or the **N most similar to a seed** (analogue sets).

**Pipeline**
1. **Filter**: organic, neutral, single-fragment; B/Si/halogens allowed; metals excluded.
2. **Encode**: choose **Descriptors + Morgan FP**, **ChemBERTa**, or **Hybrid**.
3. **Cluster**: KMeans (auto-k via silhouette) or GMM.
4. **Select**: Greedy MaxMin diversity or nearest neighbors to a seed.
5. **Visualize**: 3D **UMAP/Isomap** embedding and **PCA** (descriptor axes).
6. **Export**: selected subset CSV/SDF + annotated library CSV.
"""
    if hasattr(st, "popover"):
        with st.popover("ℹ️ About", use_container_width=True):
            st.markdown(content)
    else:
        with st.expander("ℹ️ About", expanded=False):
            st.markdown(content)

# Flip columns so title is in wide column
c1, c2 = st.columns([6, 1])
with c1:
    st.title("Chemical Library Selector")
with c2:
    about_popover()

st.caption("Filter → Encode → Cluster → **Select** most **diverse** or **similar** compounds for scope design.")

# Utility: popover if available, else expander
def info_popover(label: str, content_md: str):
    if hasattr(st, "popover"):
        with st.popover(label, use_container_width=True):
            st.markdown(content_md)
    else:
        with st.expander(label, expanded=False):
            st.markdown(content_md)

# Demo dataset
def get_demo_df():
    demo_smiles = [
        "CCO", "CCCO", "CC(C)O", "CCOC(=O)C", "c1ccccc1", "c1ccncc1", "CCN(CC)CC",
        "CC(=O)OCC", "CCCCBr", "CCSi(CH3)3", "B(O)OCC", "CCCl", "CC(C)Cl",
        "CC(C)C(=O)O", "CC(C)CO", "CCOC(F)(F)F", "CCCC(=O)N", "CCc1ccccc1",
        "C1=CC(=O)NC(=O)N1", "CC(C)C(C)O", "C[C@H](O)C(=O)O", "CC(C)N", "CCOCC",
        "COc1ccccc1", "COc1ncccc1", "CC(C)Br"
    ]
    return pd.DataFrame({"SMILES": demo_smiles})

# ===================== Sidebar: Controls & Help =====================
st.sidebar.header("Preset workflows")
colA, colB, colC = st.sidebar.columns(3)
if colA.button("Diverse 20"):
    st.session_state['selection_mode'] = "Select most DIVERSE N"
    st.session_state['N_pick'] = 20
if colB.button("Seed 12"):
    st.session_state['selection_mode'] = "Select N most SIMILAR to a seed"
    st.session_state['N_pick'] = 12
if colC.button("Explore"):
    st.session_state['selection_mode'] = "Select most DIVERSE N"
    st.session_state['N_pick'] = 30

st.sidebar.header("0) Data")
use_demo = st.sidebar.checkbox("Load demo dataset", value=False, help="Try the app without uploading a file.")

st.sidebar.header("1) Feature set & metrics")
feat_mode = st.sidebar.selectbox(
    "Feature set",
    ["Descriptors + Morgan FP (baseline)", "ChemBERTa embeddings", "ChemBERTa + Descriptors"],
    index=0,
    help="How molecules are represented numerically for distances & clustering."
)

metric_help_text = (
    "• **auto** → picks a sensible default (Jaccard for fingerprints, Cosine for ChemBERTa, Euclidean otherwise).\\n"
    "• **euclidean** → straight-line distance in standardized continuous spaces (descriptors/hybrids).\\n"
    "• **cosine** → compares direction of vectors; great for transformer embeddings.\\n"
    "• **jaccard** → overlap of on-bits; ideal for binary fingerprints (ECFP/Morgan)."
)

metric_default = {"Descriptors + Morgan FP (baseline)": "auto",
                  "ChemBERTa embeddings": "cosine",
                  "ChemBERTa + Descriptors": "euclidean"}[feat_mode]
metric_choice = st.sidebar.selectbox(
    "Distance metric (for selection/clustering)",
    ["auto","euclidean","cosine","jaccard"],
    index=["auto","euclidean","cosine","jaccard"].index(metric_default),
    help=metric_help_text
)

info_popover("ℹ️ Feature Set & Metric (when to use what)",
             r"""
**Feature sets**
1) **Descriptors + Morgan FP (baseline)**  
   - Use for **fast**, **interpretable** diversity; strong default for scope coverage.  
   - Metric: **Jaccard** (bitwise) or **Euclidean** if you rely on descriptors primarily.

2) **ChemBERTa embeddings**  
   - Use for **semantic similarity** and subtle context; good for analogue sets.  
   - Metric: **Cosine** (default).

3) **ChemBERTa + Descriptors (hybrid)**  
   - Use when you want **richness + interpretability**; strong for clustering + modeling.  
   - Metric: **Euclidean** on standardized features.
             """
)

st.sidebar.markdown("### Embedding (visualization only)")
embed_view = st.sidebar.selectbox(
    "Embedding view",
    ["UMAP (if installed)", "Isomap", "Both"],
    index=0,
    help="Choose which embedding(s) to render for the 3D feature-space view."
)
n_neighbors = st.sidebar.slider("Neighbors (UMAP/Isomap)", 5, 100, 30, 1, help="Controls local vs global structure in the 3D embedding.")
min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 0.99, 0.10, 0.01, help="Smaller = tighter clusters; larger = more spread out (UMAP only).")

st.sidebar.markdown("### Fingerprints (baseline only)")
fp_radius = st.sidebar.slider("Morgan radius", 1, 3, 2, 1, help="Neighborhood radius for ECFP; radius=2 is common.")
fp_bits = st.sidebar.select_slider("Bits", options=[256, 512, 1024, 2048], value=1024,
                                   help="Length of fingerprint; longer = fewer collisions (higher memory).")

st.sidebar.header("2) Clustering options")
cluster_algo = st.sidebar.selectbox(
    "Algorithm", ["KMeans", "GMM"], index=0,
    help="**KMeans**: spherical clusters. **GMM**: ellipsoidal clusters with covariance (soft assignments)."
)
auto_k = st.sidebar.checkbox("Auto-select k (KMeans, silhouette)", value=True,
                             help="Searches k=2..12 and picks the highest **silhouette** score.")
k_clusters = st.sidebar.slider("k (if not auto)", 2, 12, 4, 1, help="Number of clusters when auto-select is off.")

info_popover("ℹ️ Clustering help (how to choose)",
             r"""
**KMeans**: fast, robust; good default on standardized continuous spaces.  
**GMM**: allows anisotropic/overlapping clusters; gives posterior probabilities.
             """
)

st.sidebar.header("3) Selection task")
selection_mode = st.sidebar.selectbox(
    "Goal", ["Select most DIVERSE N", "Select N most SIMILAR to a seed"],
    key='selection_mode',
    help="DIVERSE: maximize spread (coverage). SIMILAR: nearest neighbors to a seed."
)
N_pick = st.sidebar.slider("N (number to select)", 3, 64, st.session_state.get('N_pick', 12), 1, help="How many compounds to pick.")
seed_smiles = st.sidebar.text_input("Seed SMILES (for 'similar to a seed')", value="", help="Must be present in the filtered set (exact SMILES match).")

info_popover("ℹ️ Selection strategies (diverse vs similar)",
             r"""
**Most DIVERSE N** → Greedy **MaxMin** (farthest-point sampling): broadest coverage of the space.  
**Most SIMILAR to a seed** → **N nearest neighbors** under the chosen distance metric (plus the seed).
             """
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Save this .py while running; Streamlit auto-reloads with your changes.")

# ===================== Chemistry helpers =====================
METAL_Z = set([
    3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,
    31,37,38,39,40,41,42,43,44,45,46,47,48,
    49,55,56,57,72,73,74,75,76,77,78,79,80,
    81,87,88,89
]) | set(range(58,72)) | set(range(90,104))

DESC_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBA": Lipinski.NumHAcceptors,
    "HBD": Lipinski.NumHDonors,
    "RotB": Lipinski.NumRotatableBonds,
    "Fsp3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "AromaticProportion": lambda m: (rdMolDescriptors.CalcNumAromaticRings(m) / max(1, rdMolDescriptors.CalcNumRings(m))),
}

def parse_mol(smi: str):
    if not isinstance(smi, str) or len(smi.strip()) < 3 or "*" in smi:
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        Chem.SanitizeMol(m, catchErrors=True)
        return m
    except Exception:
        return None

def total_formal_charge(m): return int(sum(a.GetFormalCharge() for a in m.GetAtoms()))
def contains_metal(m): return any(a.GetAtomicNum() in METAL_Z for a in m.GetAtoms())
def has_carbon(m): return any(a.GetAtomicNum()==6 for a in m.GetAtoms())
def is_multifragment(s): return "." in s if isinstance(s,str) else False

def passes_filter(smi: str) -> bool:
    if is_multifragment(smi): return False
    m = parse_mol(smi)
    if m is None: return False
    if contains_metal(m): return False
    if not has_carbon(m): return False
    if total_formal_charge(m) != 0: return False
    return True

@st.cache_data(show_spinner=False)
def filter_inventory(df, smi_col):
    keep, dropped = [], []
    for idx, smi in df[smi_col].fillna("").astype(str).items():
        if passes_filter(smi): keep.append(idx)
        else:
            m = parse_mol(smi)
            reason = "Unparsable" if m is None else (
                "Salt/multifragment" if is_multifragment(smi) else
                "Contains metal" if contains_metal(m) else
                "No carbon atoms" if not has_carbon(m) else
                "Charged (non-neutral)"
            )
            dropped.append((idx, smi, reason))
    return df.loc[keep].reset_index(drop=True), pd.DataFrame(dropped, columns=["row","SMILES","reason"])

@st.cache_data(show_spinner=False)
def compute_descriptors(smiles_list):
    rows, mols = [], []
    for smi in smiles_list:
        m = parse_mol(smi); mols.append(m)
        if m is None: rows.append({k: np.nan for k in DESC_FUNCS}); continue
        row = {}
        for name, fn in DESC_FUNCS.items():
            try: row[name] = fn(m)
            except: row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows), mols

def morgan_fp_bits(mol, radius=2, nBits=1024):
    if mol is None: return np.zeros((nBits,), dtype=np.int8)
    try:
        bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.int8)
        from rdkit import DataStructs as DS
        DS.ConvertToNumpyArray(bv, arr)
        return arr
    except: return np.zeros((nBits,), dtype=np.int8)

@st.cache_resource(show_spinner=False)
def load_chemberta():
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        mdl = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        return tok, mdl, None
    except Exception as e:
        return None, None, e

def chemberta_embed(smiles_list, batch_size=32):
    tok, mdl, err = load_chemberta()
    if err is not None or tok is None or mdl is None:
        st.warning(f"ChemBERTa not available ({type(err).__name__ if err else 'N/A'}). Using baseline features.")
        return None
    try:
        import torch
    except Exception as e:
        st.warning(f"PyTorch not available ({type(e).__name__}). Using baseline features.")
        return None
    mdl.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True)
            outputs = mdl(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embs.append(pooled)
    return np.vstack(embs)

def pick_metric(feat_mode, metric_choice, is_fp=False, is_chemberta=False):
    if metric_choice != "auto": return metric_choice
    if is_chemberta: return "cosine"
    if is_fp: return "jaccard"
    return "euclidean"

def farthest_point_sampling(D, k):
    n = D.shape[0]
    if k >= n: return list(range(n))
    avg = D.mean(axis=1)
    sel = [int(np.argmax(avg))]
    min_d = D[sel[0], :].copy()
    for _ in range(1, k):
        idx = int(np.argmax(min_d))
        sel.append(idx)
        min_d = np.minimum(min_d, D[idx, :])
    return sel

def nearest_neighbors(D, idx_seed, k):
    order = np.argsort(D[idx_seed, :])
    order = order[order != idx_seed]
    return order[:k].tolist()

# ===================== Data intake =====================
uploaded = None
if not use_demo:
    uploaded = st.file_uploader("Upload CSV with a SMILES column", type=["csv"], help="Provide a CSV with at least one column containing SMILES strings.")

if use_demo:
    df = get_demo_df()
    st.success("Loaded demo dataset.")
else:
    if uploaded is None:
        st.info("Upload a CSV or check 'Load demo dataset' to begin.")
        st.stop()
    df = pd.read_csv(uploaded)

st.write("Preview:", df.head())

default_smi_cols = [c for c in df.columns if c.strip().lower() in {"smiles","smile","structure","canonical_smiles"}]
smi_col = st.selectbox("SMILES column", list(df.columns), index=(df.columns.get_loc(default_smi_cols[0]) if default_smi_cols else 0),
                       help="Choose the column that contains SMILES.")

# ===== Pre-filter sliders (descriptor ranges) =====
st.subheader("Pre-filters (descriptor ranges)")
desc_preview, mols_preview = compute_descriptors(df[smi_col].fillna("").astype(str).tolist())
q = desc_preview.quantile([0.05, 0.95])
def qv(name, lo=True, fallback=(0.0, 1.0)):
    if name in q.columns:
        return float(q.loc[0.05, name]) if lo else float(q.loc[0.95, name])
    return fallback[0] if lo else fallback[1]

cA, cB, cC, cD = st.columns(4)
molwt_min, molwt_max = cA.slider("MolWt range", 0.0, max(600.0, qv("MolWt", False)+50.0), (qv("MolWt"), qv("MolWt", False)))
logp_min, logp_max = cB.slider("LogP range", -2.0, max(7.0, qv("LogP", False)+1.0), (min(0.0,qv("LogP")), qv("LogP", False)))
tpsa_min, tpsa_max = cC.slider("TPSA range", 0.0, max(200.0, qv("TPSA", False)+20.0), (qv("TPSA"), qv("TPSA", False)))
fsp3_min, fsp3_max = cD.slider("Fsp³ range", 0.0, 1.0, (max(0.0, qv("Fsp3")), min(1.0, qv("Fsp3", False))), step=0.01)

mask_desc = (
    (desc_preview["MolWt"].between(molwt_min, molwt_max, inclusive="both")) &
    (desc_preview["LogP"].between(logp_min, logp_max, inclusive="both")) &
    (desc_preview["TPSA"].between(tpsa_min, tpsa_max, inclusive="both")) &
    (desc_preview["Fsp3"].between(fsp3_min, fsp3_max, inclusive="both"))
)
df = df.loc[mask_desc].reset_index(drop=True)
st.caption(f"Pre-filter kept **{len(df)}** rows before chemistry filters.")

# ===== Chemistry filter =====
with st.spinner("Filtering inventory..."):
    filtered, dropped = filter_inventory(df, smi_col)
kept_n, drop_n = len(filtered), len(dropped)

st.markdown(
    f"""
<div style="margin-top:-10px; margin-bottom:10px;">
  <span style="background:#eef3ff; padding:6px 10px; border-radius:12px; margin-right:6px;">Kept: <b>{kept_n}</b></span>
  <span style="background:#fff0e6; padding:6px 10px; border-radius:12px; margin-right:6px;">Dropped: <b>{drop_n}</b></span>
</div>
""",
    unsafe_allow_html=True
)

with st.expander("Dropped entries (reasons)"):
    st.dataframe(dropped, use_container_width=True, height=210)

if kept_n==0:
    st.warning("No molecules passed the filters.")
    st.stop()

# ===== Encoding =====
with st.spinner("Encoding features..."):
    desc_df, mols = compute_descriptors(filtered[smi_col].tolist())
    desc_cols = list(DESC_FUNCS.keys())
    X_desc = desc_df.astype(float).fillna(desc_df.median(numeric_only=True))
    scaler_desc = StandardScaler()
    X_desc_s = scaler_desc.fit_transform(X_desc)

    # fingerprints
    fps = np.vstack([morgan_fp_bits(m, radius=fp_radius, nBits=fp_bits) for m in mols]).astype(float)
    # chemberta
    X_chemberta = None
    is_chemberta = ("ChemBERTa" in feat_mode)
    if is_chemberta:
        X_chemberta = chemberta_embed(filtered[smi_col].tolist())

    if feat_mode == "Descriptors + Morgan FP (baseline)":
        X = np.hstack([X_desc_s, fps]); metric = pick_metric(feat_mode, metric_choice, is_fp=True, is_chemberta=False)
    elif feat_mode == "ChemBERTa embeddings":
        if X_chemberta is None:
            X = np.hstack([X_desc_s, fps]); metric = pick_metric(feat_mode, metric_choice, is_fp=True, is_chemberta=False)
            st.info("Using baseline features instead (ChemBERTa unavailable).")
        else:
            X = StandardScaler().fit_transform(X_chemberta); metric = pick_metric(feat_mode, metric_choice, is_fp=False, is_chemberta=True)
    else:
        if X_chemberta is None:
            X = np.hstack([X_desc_s, fps]); metric = pick_metric(feat_mode, metric_choice, is_fp=True, is_chemberta=False)
            st.info("Using baseline features instead (ChemBERTa unavailable).")
        else:
            X = np.hstack([StandardScaler().fit_transform(X_chemberta), X_desc_s]); metric = pick_metric(feat_mode, metric_choice, is_fp=False, is_chemberta=False)

# ===== Distances =====
with st.spinner(f"Computing pairwise distances ({metric})..."):
    D = pairwise_distances(X, metric=metric)

# ===== Embedding(s) =====
with st.spinner("Embedding to 3D for visualization..."):
    embed_umap, embed_iso = None, None
    umap_error = None
    if embed_view in ("UMAP (if installed)", "Both"):
        try:
            import umap
            reducer = umap.UMAP(n_components=3, n_neighbors=int(n_neighbors), min_dist=float(min_dist),
                                metric="euclidean", random_state=42)
            embed_umap = reducer.fit_transform(X)
        except Exception as e:
            umap_error = e
            if embed_view == "UMAP (if installed)":
                st.warning(f"UMAP unavailable: {e.__class__.__name__}. Falling back to Isomap.")
    if embed_view in ("Isomap", "Both") or (embed_umap is None):
        iso = Isomap(n_components=3, n_neighbors=int(n_neighbors))
        embed_iso = iso.fit_transform(X)

# ===== Clustering =====
#Improvments to make... use BIC/AIC for GMM scoring or HDBSCAN
#Upper limit of 12 clusers is arbitrary use an adaptive limit
# For metrics try Tanimoto distance
# For enhanced intrepertability computer cluser centroids in chemical space and visualize them
with st.spinner("Clustering..."):
    if auto_k and cluster_algo == "KMeans":
        scores = [] #empty list to be populated by scores for each cluster
        for k in range(2, min(12, max(3, len(filtered)//2))): # min of two cluster and max of 12 also ensures that no more than half the data is in one cluster
            km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X) # cluster into K clusters minimizing distance within the cluster... do this 20 times
            s = silhouette_score(X, km.labels_, metric=metric if metric!="jaccard" else "euclidean") # score how well seperated the clusters. Uses the users chosen metric
            scores.append((k, s)) # ex
        best_k = max(scores, key=lambda t: t[1])[0] if scores else 3 # finds the cluseting aproach with the highest score... if no score use k = 3
        km = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit(X) # refir final Kmeans with optimal K
        labels = km.labels_; cluster_info = f"KMeans (k={best_k})"
    elif cluster_algo == "KMeans": # used in the case of manual clustering
        km = KMeans(n_clusters=int(k_clusters), n_init=20, random_state=42).fit(X)
        labels = km.labels_; cluster_info = f"KMeans (k={km.n_clusters})"
    else: #runs and Gaussian Matrix Model
        k = int(k_clusters)
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(X)
        labels = gmm.predict(X); cluster_info = f"GMM (k={k})"

st.markdown(
    f"""
<div style="margin-top:-6px; margin-bottom:12px;">
  <span style="background:#e8fff2; padding:6px 10px; border-radius:12px; margin-right:6px;">Clustering: <b>{cluster_info}</b></span>
  <span style="background:#f0f4f7; padding:6px 10px; border-radius:12px; margin-right:6px;">Metric: <b>{metric}</b></span>
</div>
""",
    unsafe_allow_html=True
)

# ===== Selection =====
st.subheader("Selection")
if selection_mode == "Select most DIVERSE N":
    sel_idx = farthest_point_sampling(D, int(N_pick))
    strategy_info = "Greedy MaxMin diversity (farthest-point sampling)"
else:
    if seed_smiles.strip():
        try:
            idx_seed = filtered[smi_col].tolist().index(seed_smiles.strip())
        except ValueError:
            st.warning("Seed SMILES not found in filtered set; falling back to first row.")
            idx_seed = 0
    else:
        idx_seed = 0
    sel_idx = [idx_seed] + nearest_neighbors(D, idx_seed, int(N_pick))
    strategy_info = f"N-nearest neighbors to seed index {idx_seed}"

st.info(f"Selection strategy: **{strategy_info}**  |  Distance metric: **{metric}**  |  Clustering: **{cluster_info}**")

# ===== PCA for interpretable axes =====
# Fit full PCA up to 10 PCs (or as many as features)
max_pcs = min(10, X_desc_s.shape[1]) #PCA on desciptor space into XX axes
pca = PCA(n_components=max_pcs, random_state=42).fit(X_desc_s) #Creates PCA with random seed.. finds covaraince matrix and eigenvecors/ values
pc_scores_full = pca.transform(X_desc_s) #data transfored by PCA matrix
expl_var_full = pca.explained_variance_ratio_  # return an array showing variance explained by each descriptor

# Sidebar control: pick any 3 PCs for 3D plot
pc_options = [f"PC{i}" for i in range(1, max_pcs+1)]
default_pcs = ["PC1", "PC2", "PC3"] if len(pc_options) >= 3 else pc_options[:3]
chosen_pcs = st.multiselect("Choose PCs for 3D PCA view (up to PC10)", pc_options, default=default_pcs, max_selections=3, help="Pick exactly 3 PCs to visualize.")
if len(chosen_pcs) != 3:
    st.warning("Please choose exactly **3** PCs.")
    st.stop()

pc_indices = [int(p[2:]) - 1 for p in chosen_pcs]  # zero-based
pc_labels = [f"Descriptor_{p}" for p in chosen_pcs]
pc_vars = [expl_var_full[i] for i in pc_indices]
pc_cum = sum(pc_vars)

# Build PCA plot dataframe for chosen PCs
df_pca = pd.DataFrame({
    "SMILES": filtered[smi_col].tolist(),
    "cluster": labels.astype(int),
    "selected": False
})
df_pca.loc[sel_idx, "selected"] = True
df_pca[pc_labels[0]] = pc_scores_full[:, pc_indices[0]]
df_pca[pc_labels[1]] = pc_scores_full[:, pc_indices[1]]
df_pca[pc_labels[2]] = pc_scores_full[:, pc_indices[2]]

# ===== Plotly 3D PCA (chosen PCs) =====
st.markdown("### PCA — Descriptor space")
fig_pca = px.scatter_3d(
    df_pca, x=pc_labels[0], y=pc_labels[1], z=pc_labels[2],
    color="cluster",
    title="PCA (Descriptor space)",
    opacity=0.85
)
fig_pca.update_traces(marker=dict(size=4))
fig_pca.update_layout(scene = dict(
    xaxis_title=f"{pc_labels[0]} ({pc_vars[0]*100:.1f}%)",
    yaxis_title=f"{pc_labels[1]} ({pc_vars[1]*100:.1f}%)",
    zaxis_title=f"{pc_labels[2]} ({pc_vars[2]*100:.1f}%)"
), height=560)
st.plotly_chart(fig_pca, use_container_width=True)
st.caption(f"Explained variance — {chosen_pcs[0]}: {pc_vars[0]*100:.1f}%, {chosen_pcs[1]}: {pc_vars[1]*100:.1f}%, {chosen_pcs[2]}: {pc_vars[2]*100:.1f}%.  **Cumulative:** {pc_cum*100:.1f}%")

try:
    imgp = fig_pca.to_image(format="png", scale=2)
    st.download_button("Download PCA figure (PNG)", imgp, file_name="pca_descriptors.png")
except Exception:
    st.info("To enable PNG downloads, install kaleido: `pip install -U kaleido`")

# PCA descriptor glossary
pca_desc_md = r"""
**PCA descriptor set** (standardized):
- **MolWt**, **LogP**, **TPSA**, **HBA**, **HBD**, **RotB**, **Fsp3**, **HeavyAtomCount**, **RingCount**, **AromaticProportion**
"""
if hasattr(st, "popover"):
    with st.popover("ℹ️ PCA descriptor glossary", use_container_width=True):
        st.markdown(pca_desc_md)
else:
    with st.expander("ℹ️ PCA descriptor glossary", expanded=False):
        st.markdown(pca_desc_md)

# === PCA loadings (identities of PCs) ===
pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=desc_cols,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)
st.markdown("#### PCA loadings (descriptor contributions)")
st.dataframe(pca_loadings.style.format("{:.3f}"), use_container_width=True)
buf_load = io.StringIO(); pca_loadings.to_csv(buf_load)
st.download_button("Download PCA loadings (CSV)", buf_load.getvalue(), file_name="pca_loadings.csv")

# === Scree plot (full variance profile) ===
fig_scree = px.bar(x=[f"PC{i+1}" for i in range(len(expl_var_full))],
                   y=expl_var_full * 100,
                   labels={"x":"Principal Component","y":"% Variance Explained"},
                   title=f"PCA Scree Plot — cumulative PC1–3: {expl_var_full[:3].sum()*100:.1f}%")
fig_scree.update_layout(height=340)
st.plotly_chart(fig_scree, use_container_width=True)

# ===== 3D Embedding views (feature space) =====
# (Kept unchanged below)
if 'embed_umap' not in locals():
    embed_umap = None
if 'embed_iso' not in locals():
    embed_iso = None

st.markdown("### 3D Views — Feature space")
if embed_view == "Both":
    c1, c2 = st.columns(2)
    with c1:
        if embed_umap is not None:
            st.markdown("#### Embedding: **UMAP**")
            df_embed_umap = pd.DataFrame({
                "SMILES": filtered[smi_col].tolist(),
                "cluster": labels.astype(int),
                "selected": False,
                "UMAP_1": embed_umap[:,0],
                "UMAP_2": embed_umap[:,1],
                "UMAP_3": embed_umap[:,2],
            })
            df_embed_umap.loc[sel_idx, "selected"] = True
            fig1 = px.scatter_3d(df_embed_umap, x="UMAP_1", y="UMAP_2", z="UMAP_3",
                                 color="cluster", symbol="selected",
                                 title="UMAP Embedding (feature space)", opacity=0.8)
            fig1.update_traces(marker=dict(size=4))
            fig1.update_layout(height=520)
            st.plotly_chart(fig1, use_container_width=True)
            try:
                img1 = fig1.to_image(format="png", scale=2)
                st.download_button("Download UMAP figure (PNG)", img1, file_name="embedding_umap.png")
            except Exception:
                st.info("To enable PNG downloads, install kaleido: `pip install -U kaleido`")
        else:
            st.warning("UMAP unavailable.")
    with c2:
        if embed_iso is not None:
            st.markdown("#### Embedding: **Isomap**")
            df_embed_iso = pd.DataFrame({
                "SMILES": filtered[smi_col].tolist(),
                "cluster": labels.astype(int),
                "selected": False,
                "Isomap_1": embed_iso[:,0],
                "Isomap_2": embed_iso[:,1],
                "Isomap_3": embed_iso[:,2],
            })
            df_embed_iso.loc[sel_idx, "selected"] = True
            fig2 = px.scatter_3d(df_embed_iso, x="Isomap_1", y="Isomap_2", z="Isomap_3",
                                 color="cluster", symbol="selected",
                                 title="Isomap Embedding (feature space)", opacity=0.8)
            fig2.update_traces(marker=dict(size=4))
            fig2.update_layout(height=520)
            st.plotly_chart(fig2, use_container_width=True)
            try:
                img2 = fig2.to_image(format="png", scale=2)
                st.download_button("Download Isomap figure (PNG)", img2, file_name="embedding_isomap.png")
            except Exception:
                st.info("To enable PNG downloads, install kaleido: `pip install -U kaleido`")
else:
    if embed_view == "UMAP (if installed)" and embed_umap is not None:
        st.markdown("#### Embedding: **UMAP**")
        df_embed_umap = pd.DataFrame({
            "SMILES": filtered[smi_col].tolist(),
            "cluster": labels.astype(int),
            "selected": False,
            "UMAP_1": embed_umap[:,0],
            "UMAP_2": embed_umap[:,1],
            "UMAP_3": embed_umap[:,2],
        })
        df_embed_umap.loc[sel_idx, "selected"] = True
        fig = px.scatter_3d(df_embed_umap, x="UMAP_1", y="UMAP_2", z="UMAP_3",
                            color="cluster", symbol="selected",
                            title="UMAP Embedding (feature space)", opacity=0.8)
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)
        try:
            img = fig.to_image(format="png", scale=2)
            st.download_button("Download UMAP figure (PNG)", img, file_name="embedding_umap.png")
        except Exception:
            st.info("To enable PNG downloads, install kaleido: `pip install -U kaleido`")
    else:
        st.markdown("#### Embedding: **Isomap**")
        df_embed_iso = pd.DataFrame({
            "SMILES": filtered[smi_col].tolist(),
            "cluster": labels.astype(int),
            "selected": False,
            "Isomap_1": embed_iso[:,0],
            "Isomap_2": embed_iso[:,1],
            "Isomap_3": embed_iso[:,2],
        })
        df_embed_iso.loc[sel_idx, "selected"] = True
        fig = px.scatter_3d(df_embed_iso, x="Isomap_1", y="Isomap_2", z="Isomap_3",
                            color="cluster", symbol="selected",
                            title="Isomap Embedding (feature space)", opacity=0.8)
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)
        try:
            img = fig.to_image(format="png", scale=2)
            st.download_button("Download Isomap figure (PNG)", img, file_name="embedding_isomap.png")
        except Exception:
            st.info("To enable PNG downloads, install kaleido: `pip install -U kaleido`")

# ===== SVM feature importance (bar chart) =====
st.markdown("### SVM — Descriptor feature importances")
svm = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=42).fit(X_desc_s, labels) #train linear SVM model on labels from clusering
coef_abs = np.abs(svm.coef_) # get absolute coefficent magnitudes
importances_vec = coef_abs.mean(axis=0) if coef_abs.ndim == 2 else coef_abs # average across classes
svm_importances = pd.DataFrame({"descriptor": list(desc_cols), "importance": importances_vec}).sort_values("importance", ascending=False) #create df for ranked features
fig_svm = px.bar(svm_importances.head(12), x="descriptor", y="importance", title="SVM (Linear) — top descriptor weights")
fig_svm.update_layout(height=420)
st.plotly_chart(fig_svm, use_container_width=True)
try:
    imgs = fig_svm.to_image(format="png", scale=2)
    st.download_button("Download SVM figure (PNG)", imgs, file_name="svm_descriptor_importances.png")
except Exception:
    st.info("To enable PNG downloads, install kaleido: `pip install -U kaleido`")

# ===== Downloads for plot data =====
st.markdown("### Download plot data as CSV")
# PCA plot data (chosen PCs)
buf_pca = io.StringIO(); df_pca.to_csv(buf_pca, index=False)
st.download_button("Download PCA plot data (CSV)", buf_pca.getvalue(), file_name="pca_plot_data.csv")

# Embeddings (if present)
if 'df_embed_umap' in locals():
    buf_embed_umap = io.StringIO(); df_embed_umap.to_csv(buf_embed_umap, index=False)
    st.download_button("Download UMAP plot data (CSV)", buf_embed_umap.getvalue(), file_name="embedding_umap_plot_data.csv")
if 'df_embed_iso' in locals():
    buf_embed_iso = io.StringIO(); df_embed_iso.to_csv(buf_embed_iso, index=False)
    st.download_button("Download Isomap plot data (CSV)", buf_embed_iso.getvalue(), file_name="embedding_isomap_plot_data.csv")

# SVM importances
buf_svm = io.StringIO(); svm_importances.to_csv(buf_svm, index=False)
st.download_button("Download SVM descriptor importances (CSV)", buf_svm.getvalue(), file_name="svm_descriptor_importances.csv")

# ===== Exports: selection + SDF + annotated library =====
st.markdown("### Download selection & annotated library")
sel_table = filtered.iloc[sel_idx, :].copy()
sel_table.insert(0, "selection_rank", list(range(1, len(sel_idx)+1)))
sel_csv = io.StringIO(); sel_table.to_csv(sel_csv, index=False)
st.download_button("Download selected compounds (CSV)", sel_csv.getvalue(), file_name="selected_compounds.csv", type="primary")

# SDF of selected (StringIO -> UTF-8 bytes)
sdf_text = io.StringIO()
w = SDWriter(sdf_text)
for smi in sel_table[smi_col].tolist():
    m = Chem.MolFromSmiles(smi)
    if m: w.write(m)
w.close()
st.download_button(
    "Download selected compounds (SDF)",
    data=sdf_text.getvalue().encode("utf-8"),
    file_name="selected_compounds.sdf",
    mime="chemical/x-mdl-sdfile",
)

annotated = filtered.copy()
annotated["cluster"] = labels
# Also add chosen PC scores to annotated table
annotated[pc_labels[0]] = df_pca[pc_labels[0]]
annotated[pc_labels[1]] = df_pca[pc_labels[1]]
annotated[pc_labels[2]] = df_pca[pc_labels[2]]
ann_csv = io.StringIO(); annotated.to_csv(ann_csv, index=False)
st.download_button("Download annotated library (CSV)", ann_csv.getvalue(), file_name="annotated_library.csv")
