from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import mygene
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app)

# Global cache to store processed data and image
cached_result_df = None
cached_base64_image = None

def load_and_prepare_data():
    global cached_result_df, cached_base64_image

    if cached_base64_image is not None:
        return  # Already processed

    # Load the data
    df = pd.read_csv('GSE19804_series_matrix.txt', sep="\t", comment='!', index_col=0)
    df = df.transpose()

    # Keep only float columns (gene expression)
    df = df.select_dtypes(include='float64')

    # Assign sample labels (first 60 = tumor, next 60 = normal)
    df['label'] = ['Tumor'] * 60 + ['Normal'] * 60

    # Split data
    tumor = df[df['label'] == 'Tumor'].drop(columns=['label'])
    normal = df[df['label'] == 'Normal'].drop(columns=['label'])

    # Perform t-test for each gene
    results = [(gene, *ttest_ind(tumor[gene], normal[gene])) for gene in tumor.columns]
    result_df = pd.DataFrame(results, columns=['Gene', 'T-statistic', 'P-value']).sort_values('P-value')

    # Annotate genes using MyGene
    mg = mygene.MyGeneInfo()
    probes = result_df['Gene'].tolist()
    annotations = mg.querymany(probes, scopes='reporter', fields='symbol', species='human')

    annot_df = pd.DataFrame(annotations)
    annot_df = annot_df[annot_df['symbol'].notna()]
    probe_to_symbol = dict(zip(annot_df['query'], annot_df['symbol']))
    result_df['GeneSymbol'] = result_df['Gene'].map(probe_to_symbol)
    result_df = result_df.dropna(subset=['GeneSymbol'])

    # Generate the volcano plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        result_df['T-statistic'],
        -np.log10(result_df['P-value']),
        c=(result_df['P-value'] < 0.05),
        cmap='coolwarm',
        alpha=0.7
    )
    plt.xlabel('T-statistic')
    plt.ylabel('-log10(p-value)')
    plt.title('Volcano Plot of Differential Gene Expression')
    plt.axhline(-np.log10(0.05), color='gray', linestyle='--')

    # Save to memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Cache the result
    cached_result_df = result_df
    cached_base64_image = image_base64

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Flask!"})

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.json
    return jsonify({"received": data})

@app.route('/api/cancer', methods=['GET'])
def get_cancer_graphs():
    load_and_prepare_data()
    return jsonify({"plot": cached_base64_image})

if __name__ == '__main__':
    print("⏳ Preprocessing gene expression data...")
    load_and_prepare_data()
    print("✅ Data loaded and cached.")
    app.run(debug=True, port=5000)
