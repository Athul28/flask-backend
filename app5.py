from flask import Flask, jsonify
from flask_cors import CORS
import GEOparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap.umap_ as umap
import io
import base64

app = Flask(__name__)
CORS(app)

# Global cache for plots
plot_cache = {}

def load_and_process_data():
    try:
        print("Loading GEO data...")
        gse = GEOparse.get_GEO("GSE96058", destdir=".")
        expr_df = pd.read_csv("GSE96058_transcript_expression_3273_samples_and_136_replicates.csv.gz",
                              compression='gzip', index_col=0)

        replicate_cols = [col for col in expr_df.columns if 'repl' in col]
        main_cols = [col for col in expr_df.columns if 'repl' not in col]
        expr_df_main = expr_df[main_cols]

        expr_df_log = np.log2(expr_df_main + 1)
        expr_df_log = expr_df_log.replace([np.inf, -np.inf], np.nan).fillna(0)

        X = expr_df_log.T  # Transpose for samples as rows
        return X

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def generate_pca_plot(X):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.6, edgecolor=None)
    plt.title("PCA of Breast Cancer Patients - GSE96058")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    return fig_to_base64(plt)


def generate_kmeans_plot(X):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2")
    plt.title("K-Means Clustering of Patients")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    return fig_to_base64(plt)


def generate_tsne_plot(X):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1])
    plt.title("t-SNE of Breast Cancer Patients - GSE96058")
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")
    plt.grid(True)
    return fig_to_base64(plt)


def generate_umap_plot(X):
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1])
    plt.title("UMAP of Breast Cancer Patients - GSE96058")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    return fig_to_base64(plt)


def fig_to_base64(plt_obj):
    buf = io.BytesIO()
    plt_obj.savefig(buf, format="png")
    plt_obj.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/pca-plot')
def pca_plot():
    return jsonify({"image": plot_cache.get("pca")})


@app.route('/kmeans-plot')
def kmeans_plot():
    return jsonify({"image": plot_cache.get("kmeans")})


@app.route('/tsne-plot')
def tsne_plot():
    return jsonify({"image": plot_cache.get("tsne")})


@app.route('/umap-plot')
def umap_plot():
    return jsonify({"image": plot_cache.get("umap")})


if __name__ == '__main__':
    print("Starting server and generating plots...")
    X = load_and_process_data()

    if X is not None:
        plot_cache["pca"] = generate_pca_plot(X)
        plot_cache["kmeans"] = generate_kmeans_plot(X)
        plot_cache["tsne"] = generate_tsne_plot(X)
        plot_cache["umap"] = generate_umap_plot(X)
        print("All plots cached and ready!")
    else:
        print("Failed to load data. Check errors above.")

    app.run(debug=True, port=5000)
