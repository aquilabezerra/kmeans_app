import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="K-Means Clustering — Online Shoppers", layout="wide")

# -------------------------
# Helpers (cached where appropriate)
# -------------------------
@st.cache_data
def load_default_dataset():
    df = pd.read_csv('online_shoppers_intention.csv')
    return df

@st.cache_data
def preprocess_numeric(df):
    df_clean = df.copy()
    # remove non-numeric columns
    num = df_clean.select_dtypes(include=["number"])
    num = num.dropna()  # keep only complete numeric rows by default
    return num

@st.cache_data
def scale_data(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs

@st.cache_data
def compute_kmeans(Xs, k, random_state=42):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs)
    return km, labels

def plot_elbow(Xs, k_max=10):
    wcss = []
    ks = list(range(1, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
        km.fit(Xs)
        wcss.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(ks, wcss, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Método do Cotovelo")
    ax.grid(True)
    st.pyplot(fig)

def plot_pca_scatter(Xs, labels):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xs)
    fig, ax = plt.subplots()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=0.8)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Visualização em PCA (2 componentes)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# -------------------------
# App UI
# -------------------------
st.title("🔵 K-Means Clustering — Projeto pronto (Online Shoppers)")

with st.sidebar:
    st.header("Configuração")
    use_default = st.radio("Dataset", ("Usar dataset padrão (Online Shoppers)", "Carregar meu CSV"), index=0)
    if use_default == "Carregar meu CSV":
        uploaded_file = st.file_uploader("Escolha um arquivo .csv", type=["csv"])
        sample_rows = st.number_input("Linhas de amostra para mostrar (preview)", min_value=5, max_value=1000, value=10, step=5)
    else:
        uploaded_file = None
        uploaded_file = None
        sample_rows = st.number_input("Linhas de amostra para mostrar (preview)", min_value=5, max_value=1000, value=10, step=5)
    st.markdown("---")
    st.header("K-Means")
    k_default = st.slider("Número de clusters (k)", 2, 12, 4)
    show_elbow = st.checkbox("Mostrar gráfico do cotovelo", value=True)
    show_silhouette = st.checkbox("Mostrar silhouette scores (k=2..10)", value=True)
    st.markdown("---")
    st.header("Pré-processamento")
    drop_na = st.checkbox("Remover linhas com valores ausentes (aplicado antes de selecionar colunas numéricas)", value=True)
    st.markdown("---")
    st.header("Export")
    enable_download = st.checkbox("Habilitar download do CSV com clusters", value=True)

st.markdown("### 1) Carregar e visualizar dados")
if uploaded_file is None and use_default.startswith("Usar"):
    df = load_default_dataset()
    st.success("Dataset padrão carregado (Online Shoppers Intention).")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Arquivo carregado com sucesso.")
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()
else:
    st.info("Escolha um dataset na barra lateral.")
    st.stop()

st.write("Tamanho do dataset original:", df.shape)
st.dataframe(df.head(sample_rows))

st.markdown("### 2) Pré-processamento automático (apenas colunas numéricas)")
df_proc = df.copy()
if drop_na:
    df_proc = df_proc.dropna()
num = df_proc.select_dtypes(include=["number"])
if num.shape[1] == 0:
    st.error("Não foram encontradas colunas numéricas no dataset. Faça upload de outro CSV ou converta colunas para numéricas.")
    st.stop()

st.write(f"Colunas numéricas detectadas ({num.shape[1]}): {list(num.columns)}")
st.dataframe(num.head(5))

# scaling
Xs = scale_data(num)

st.markdown("### 3) Escolher k e treinar K-Means")
if show_elbow:
    plot_elbow(Xs, k_max=12)

if show_silhouette:
    st.markdown("**Silhouette scores** (k e respectivo score):")
    sil_lines = []
    for kk in range(2, 11):
        km_tmp = KMeans(n_clusters=kk, init="k-means++", n_init=5, random_state=42)
        labels_tmp = km_tmp.fit_predict(Xs)
        try:
            sc = silhouette_score(Xs, labels_tmp)
        except Exception:
            sc = np.nan
        sil_lines.append((kk, sc))
    sil_df = pd.DataFrame(sil_lines, columns=["k", "silhouette_score"])
    st.table(sil_df)

k = k_default
km, labels = compute_kmeans(Xs, k)
st.success(f"KMeans treinado com k = {k}")

# attach clusters to original (filtered) numeric dataframe: align shapes
# note: after dropna we already reduced dataset; we attached 'num' derived from df_proc
df_clusters = num.copy()
df_clusters = df_clusters.reset_index(drop=True)
df_clusters["cluster"] = labels

st.markdown("### 4) Análise dos clusters")
st.write("Contagem por cluster:")
st.write(df_clusters["cluster"].value_counts().sort_index())

st.write("Médias por cluster (para colunas numéricas):")
st.dataframe(df_clusters.groupby("cluster").mean().round(4))

plot_pca_scatter(Xs, labels)

st.markdown("#### Centros dos clusters (no espaço das features escaladas)")
centers = km.cluster_centers_
centers_df = pd.DataFrame(centers, columns=num.columns)
st.dataframe(centers_df)

# Option to show original rows with cluster label (if user wants)
if st.checkbox("Mostrar linhas originais com rótulo de cluster (após remoção de NaNs)", value=False):
    # we need to align indexes: df_proc numeric rows -> df_clusters
    df_proc_numeric = df_proc.reset_index(drop=True)
    # merge numeric columns back into original rows by index (works because we dropped rows uniformly earlier)
    merged = df_proc_numeric.copy()
    # ensure numeric columns present
    merged = merged.loc[:, list(num.columns) + [c for c in df_proc_numeric.columns if c not in num.columns]]
    merged["cluster"] = df_clusters["cluster"]
    st.dataframe(merged.head(200))

# Download results
if enable_download:
    result_df = df_proc.reset_index(drop=True).copy()
    result_df["cluster"] = df_clusters["cluster"].values
    csv_bytes = df_to_csv_bytes(result_df)
    st.download_button("📥 Baixar CSV com cluster", data=csv_bytes, file_name="clustered_data.csv", mime="text/csv")

st.markdown("## Observações e próximos passos")
st.write("""
- Se quiser usar suas próprias colunas, carregue um CSV com as colunas numéricas de interesse.
- É recomendável analisar as features categóricas (convertendo com one-hot) se quiser incluir mais variáveis no clustering.
- Você pode ajustar `n_init`, `random_state` e o pré-processamento para testar estabilidade.
""")