from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import os

app = Flask(__name__)
CORS(app)

data_path = "GSE150910_gene-level_count_file.csv"
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

@app.route("/api/top-gene-analysis", methods=["GET"])
def analyze_top_gene():
    df = pd.read_csv(data_path, index_col=0)
    if df.shape[0] < df.shape[1]:
        df = df.transpose()

    labels = ['COVID'] * (df.shape[0] // 2) + ['Normal'] * (df.shape[0] - df.shape[0] // 2)
    df['label'] = labels

    covid_group = df[df['label'] == 'COVID'].drop(columns=['label'])
    normal_group = df[df['label'] == 'Normal'].drop(columns=['label'])

    results = []
    for gene in covid_group.columns:
        try:
            stat, p = ttest_ind(covid_group[gene], normal_group[gene])
            fold_change = covid_group[gene].mean() - normal_group[gene].mean()
            results.append((gene, stat, p, fold_change))
        except:
            continue

    result_df = pd.DataFrame(results, columns=['Gene', 'T-statistic', 'P-value', 'FoldChange'])
    result_df = result_df.sort_values('P-value')

    top_gene = result_df.iloc[0]['Gene']

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='label', y=top_gene, data=df)
    path = f"{plot_dir}/boxplot.png"
    plt.title(f'Boxplot: {top_gene}')
    plt.savefig(path)
    plt.close()

    return send_file(path, mimetype='image/png')

@app.route("/api/volcano", methods=["GET"])
def volcano_plot():
    df = pd.read_csv(data_path, index_col=0)
    if df.shape[0] < df.shape[1]:
        df = df.transpose()
    labels = ['COVID'] * (df.shape[0] // 2) + ['Normal'] * (df.shape[0] - df.shape[0] // 2)
    df['label'] = labels
    covid_group = df[df['label'] == 'COVID'].drop(columns=['label'])
    normal_group = df[df['label'] == 'Normal'].drop(columns=['label'])
    results = []
    for gene in covid_group.columns:
        try:
            stat, p = ttest_ind(covid_group[gene], normal_group[gene])
            fold_change = covid_group[gene].mean() - normal_group[gene].mean()
            results.append((gene, stat, p, fold_change))
        except:
            continue
    result_df = pd.DataFrame(results, columns=['Gene', 'T-statistic', 'P-value', 'FoldChange'])
    result_df['-log10(P)'] = -np.log10(result_df['P-value'])
    
    # Volcano Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=result_df, x='FoldChange', y='-log10(P)')
    plt.axhline(-np.log10(0.05), color='red', linestyle='--')
    plt.axvline(1, color='green', linestyle='--')
    plt.axvline(-1, color='green', linestyle='--')
    plt.title("Volcano Plot")
    plt.xlabel("Fold Change")
    plt.ylabel("-log10(P-value)")
    path = f"{plot_dir}/volcano.png"
    plt.savefig(path)
    plt.close()
    return send_file(path, mimetype='image/png')

@app.route("/api/classifier", methods=["GET"])
def train_classifier():
    df = pd.read_csv(data_path, index_col=0)
    if df.shape[0] < df.shape[1]:
        df = df.transpose()
    labels = ['COVID'] * (df.shape[0] // 2) + ['Normal'] * (df.shape[0] - df.shape[0] // 2)
    df['label'] = labels

    covid_group = df[df['label'] == 'COVID'].drop(columns=['label'])
    normal_group = df[df['label'] == 'Normal'].drop(columns=['label'])

    results = []
    for gene in covid_group.columns:
        try:
            stat, p = ttest_ind(covid_group[gene], normal_group[gene])
            fold_change = covid_group[gene].mean() - normal_group[gene].mean()
            results.append((gene, stat, p, fold_change))
        except:
            continue

    result_df = pd.DataFrame(results, columns=['Gene', 'T-statistic', 'P-value', 'FoldChange'])
    top_genes = result_df['Gene'].head(50).tolist()
    X = df[top_genes]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return jsonify({"accuracy": acc})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
