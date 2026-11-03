"""Streamlit app: Dimensionality reduction (PCA, LDA, ICA) on Breast Cancer Wisconsin data.
Based on Lab 4 - Sweezal Suares (547)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

st.set_page_config(page_title="Dimensionality Reduction Lab", page_icon="🔬", layout="wide")

PALETTE = {"malignant": "#e41a1c", "benign": "#377eb8"}


@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    diagnosis = np.array([cancer.target_names[i] for i in y])
    return df, y, diagnosis


@st.cache_data
def scale(df):
    return StandardScaler().fit_transform(df)


df, y, diagnosis = load_data()
X_scaled = scale(df)

st.title("🔬 Dimensionality Reduction: PCA, LDA & ICA")
st.caption("Breast Cancer Wisconsin (Diagnostic) dataset · Lab 4")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
page = st.sidebar.radio(
    "Section",
    ["Data Explorer", "PCA", "LDA", "ICA", "Comparison"],
)
n_pca = st.sidebar.slider("PCA components", 2, 10, 4)
n_ica = st.sidebar.slider("ICA components", 2, 10, 4)
seed = st.sidebar.number_input("Random seed (ICA)", value=42, step=1)

# ---------------- Fit models ----------------
pca_full = PCA().fit(X_scaled)
pca = PCA(n_components=n_pca).fit(X_scaled)
X_pca = pca.transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=[f"PC {i+1}" for i in range(n_pca)])
pca_df["Diagnosis"] = diagnosis

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)
lda_df = pd.DataFrame(X_lda, columns=["LDA Component 1"])
lda_df["Diagnosis"] = diagnosis

ica = FastICA(n_components=n_ica, random_state=int(seed), max_iter=1000)
X_ica = ica.fit_transform(X_scaled)
ica_df = pd.DataFrame(X_ica, columns=[f"IC {i+1}" for i in range(n_ica)])
ica_df["Diagnosis"] = diagnosis


# ---------------- Pages ----------------
if page == "Data Explorer":
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("Missing values / duplicates", f"{int(df.isnull().sum().sum())} / {int(df.duplicated().sum())}")

    tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Distributions"])
    with tab1:
        st.dataframe(df.assign(Diagnosis=diagnosis))
    with tab2:
        st.dataframe(df.describe().T)
    with tab3:
        left, right = st.columns([1, 2])
        with left:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.countplot(x=diagnosis, hue=diagnosis, palette=PALETTE, legend=False, ax=ax)
            ax.set_title("Class Distribution")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        with right:
            feat = st.selectbox("Feature histogram", df.columns)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(data=df.assign(Diagnosis=diagnosis), x=feat, hue="Diagnosis",
                         palette=PALETTE, kde=True, ax=ax)
            st.pyplot(fig)

elif page == "PCA":
    st.subheader("Principal Component Analysis")
    ratio = pca_full.explained_variance_ratio_
    cum = np.cumsum(ratio)
    n90 = int(np.argmax(cum >= 0.90) + 1)
    n95 = int(np.argmax(cum >= 0.95) + 1)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Variance kept ({n_pca} PCs)", f"{cum[n_pca-1]*100:.1f}%")
    c2.metric("PCs for 90% variance", n90)
    c3.metric("PCs for 95% variance", n95)

    k = min(10, len(ratio))
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [f"PC {i+1}" for i in range(k)]
    bars = ax.bar(names, ratio[:k] * 100, color="#377eb8", alpha=0.7, label="Individual")
    ax.plot(names, cum[:k] * 100, "--o", color="#ff7f00", label="Cumulative")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.axhline(90, color="grey", ls=":", lw=1)
    ax.axhline(95, color="grey", ls=":", lw=1)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Variance (%)")
    ax.set_title("Variance Explained by Principal Components", fontweight="bold")
    ax.legend(loc="center right")
    ax.grid(axis="y", ls="--", alpha=0.5)
    st.pyplot(fig)

    xc = st.selectbox("X axis", pca_df.columns[:-1], index=0)
    yc = st.selectbox("Y axis", pca_df.columns[:-1], index=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=pca_df, x=xc, y=yc, hue="Diagnosis", palette=PALETTE, alpha=0.7, ax=ax)
    ax.set_title(f"PCA: {xc} vs {yc}", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    st.pyplot(fig)
    with st.expander("Projected data"):
        st.dataframe(pca_df.head(20))

elif page == "LDA":
    st.subheader("Linear Discriminant Analysis (supervised)")
    st.write("LDA uses class labels, so with 2 classes it yields at most 1 component.")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, color in PALETTE.items():
        sns.histplot(lda_df[lda_df["Diagnosis"] == label], x="LDA Component 1", color=color,
                     label=label.title(), kde=True, alpha=0.6, bins=20, ax=ax)
    ax.set_title("LDA Projection: Optimal 1D Class Separation", fontweight="bold")
    ax.legend(title="Diagnosis")
    ax.grid(axis="x", ls="--", alpha=0.5)
    st.pyplot(fig)

    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.1, size=len(lda_df))
    fig, ax = plt.subplots(figsize=(9, 3))
    sns.scatterplot(x=lda_df["LDA Component 1"], y=jitter, hue=diagnosis,
                    palette=PALETTE, alpha=0.7, ax=ax)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel("Jitter")
    ax.set_title("LDA Component 1 (with jitter)", fontweight="bold")
    st.pyplot(fig)
    with st.expander("Projected data"):
        st.dataframe(lda_df.head(20))

elif page == "ICA":
    st.subheader("Independent Component Analysis")
    xc = st.selectbox("X axis", ica_df.columns[:-1], index=0)
    yc = st.selectbox("Y axis", ica_df.columns[:-1], index=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=ica_df, x=xc, y=yc, hue="Diagnosis", palette=PALETTE, alpha=0.7, ax=ax)
    ax.set_title(f"ICA: {xc} vs {yc}", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    st.pyplot(fig)

    if st.checkbox("Show pairplot (first 4 components)"):
        cols = list(ica_df.columns[:-1])[:4]
        g = sns.pairplot(ica_df, vars=cols, hue="Diagnosis", palette=PALETTE,
                         plot_kws={"alpha": 0.6, "s": 30, "edgecolor": "none"}, diag_kind="kde")
        st.pyplot(g.fig)
    with st.expander("Projected data"):
        st.dataframe(ica_df.head(20))

else:
    st.subheader("Side-by-side comparison")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.scatterplot(data=pca_df, x="PC 1", y="PC 2", hue="Diagnosis", palette=PALETTE, alpha=0.7, ax=axes[0])
    axes[0].set_title("PCA: PC 1 vs PC 2", fontweight="bold")
    rng = np.random.default_rng(42)
    sns.scatterplot(x=lda_df["LDA Component 1"], y=rng.normal(0, 0.1, len(lda_df)), hue=diagnosis,
                    palette=PALETTE, alpha=0.7, ax=axes[1])
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_title("LDA: Component 1 (jitter)", fontweight="bold")
    axes[1].set_xlabel("Linear Discriminant 1")
    axes[1].set_ylabel("Jitter")
    sns.scatterplot(data=ica_df, x="IC 1", y="IC 2", hue="Diagnosis", palette=PALETTE, alpha=0.7, ax=axes[2])
    axes[2].set_title("ICA: IC 1 vs IC 2", fontweight="bold")
    for a in axes:
        a.grid(True, ls="--", alpha=0.5)
    fig.suptitle("Comparison of Dimensionality Reduction Transformations", fontsize=15, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown(
        "- **PCA** – unsupervised, maximises variance; ignores labels.\n"
        "- **LDA** – supervised, maximises class separation (1 component for 2 classes).\n"
        "- **ICA** – unsupervised, finds statistically independent sources."
    )
