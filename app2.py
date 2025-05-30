from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import ttest_ind

app = Flask(__name__)
CORS(app)

# Caching processed data
cached_result_df = None
cached_plots = None

def load_and_analyze_covid_data():
    global cached_result_df, cached_plots

    if cached_plots is not None:
        return

    # Load the CSV
    df = pd.read_csv('GSE150910_gene-level_count_file.csv', index_col=0)

    # Transpose if genes are rows
    if df.shape[0] < df.shape[1]:
        df = df.transpose()

    # Assign labels: adjust based on actual metadata
    n_samples = df.shape[0]
    covid_samples = n_samples // 2
    normal_samples = n_samples - covid_samples
    labels = ['COVID'] * covid_samples + ['Normal'] * normal_samples
    df['label'] = labels

    # Perform t-tests
    gene_data = df.drop(columns=['label'])
    covid_group = df[df['label'] == 'COVID'].drop(columns=['label'])
    normal_group = df[df['label'] == 'Normal'].drop(columns=['label'])

    results = []
    for gene in gene_data.columns:
        try:
            stat, p = ttest_ind(covid_group[gene], normal_group[gene])
            fold_change = covid_group[gene].mean() - normal_group[gene].mean()
            results.append((gene, stat, p, fold_change))
        except:
            continue

    result_df = pd.DataFrame(results, columns=['Gene', 'T-statistic', 'P-value', 'FoldChange'])
    result_df = result_df.sort_values('P-value')
    cached_result_df = result_df

    # Analyze top gene
    top_gene = result_df.iloc[0]['Gene']

    # Generate plots
    plot_images = {}

    def plot_and_encode(kind):
        plt.figure(figsize=(8, 6))
        if kind == 'box':
            sns.boxplot(x='label', y=top_gene, data=df)
            plt.title(f'Expression of {top_gene} by Condition (Boxplot)')
        elif kind == 'violin':
            sns.violinplot(x='label', y=top_gene, data=df)
            plt.title(f'Expression of {top_gene} by Condition (Violin)')
        elif kind == 'strip':
            sns.stripplot(x='label', y=top_gene, data=df, jitter=True)
            plt.title(f'Individual Expression of {top_gene} (Strip Plot)')
        plt.ylabel("Expression Level")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    plot_images['boxplot'] = plot_and_encode('box')
    plot_images['violin'] = plot_and_encode('violin')
    plot_images['strip'] = plot_and_encode('strip')

    cached_plots = plot_images


@app.route('/api/covid', methods=['GET'])
def covid_gene_expression():
    load_and_analyze_covid_data()
    return jsonify({
        "top_genes": cached_result_df.head(10).to_dict(orient='records'),
        "plots": cached_plots
    })


if __name__ == '__main__':
    print("⏳ Processing COVID dataset...")
    load_and_analyze_covid_data()
    print("✅ COVID dataset loaded.")
    app.run(debug=True, port=5000)
