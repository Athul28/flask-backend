# === FLASK BACKEND ===

from flask import Flask, jsonify, send_file
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

DATA_PATH = 'GSE150910_gene-level_count_file.csv'
PLOT_DIR = 'static'
os.makedirs(PLOT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    if df.shape[0] < df.shape[1]:
        df = df.transpose()
    n_samples = df.shape[0]
    covid_samples = n_samples // 2
    normal_samples = n_samples - covid_samples
    labels = ['COVID'] * covid_samples + ['Normal'] * normal_samples
    df['label'] = labels
    return df

def perform_ttest(df):
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
    result_df = result_df.sort_values('P-value')
    return result_df

@app.route('/api/covid/top-genes')
def top_genes():
    df = load_data()
    result_df = perform_ttest(df)
    return jsonify(result_df[['Gene', 'P-value', 'FoldChange']].head(10).to_dict(orient='records'))

@app.route('/api/covid/volcano')
def volcano():
    df = load_data()
    result_df = perform_ttest(df)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=result_df, x='FoldChange', y='-log10(P)')
    plt.axhline(-np.log10(0.05), color='red', linestyle='--')
    plt.axvline(1, color='green', linestyle='--')
    plt.axvline(-1, color='green', linestyle='--')
    plt.title("Volcano Plot")
    plt.xlabel("Mean Expression Difference")
    plt.ylabel("-log10(P-value)")
    path = f"{PLOT_DIR}/volcano.png"
    plt.savefig(path)
    plt.close()
    return send_file(path, mimetype='image/png')

@app.route('/api/covid/pca')
def pca():
    df = load_data()
    result_df = perform_ttest(df)
    top_genes = result_df['Gene'].head(4).tolist()
    X = df[top_genes].values
    y = df['label'].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['label'] = y
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label')
    plt.title('PCA of Gene Expression')
    path = f"{PLOT_DIR}/pca.png"
    plt.savefig(path)
    plt.close()
    return send_file(path, mimetype='image/png')

@app.route('/api/covid/heatmap')
def heatmap():
    df = load_data()
    result_df = perform_ttest(df)
    top_genes = result_df['Gene'].head(6).tolist()
    corr = df[top_genes].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Gene Correlation Heatmap")
    path = f"{PLOT_DIR}/heatmap.png"
    plt.savefig(path)
    plt.close()
    return send_file(path, mimetype='image/png')

@app.route('/api/covid/classifier')
def classifier():
    df = load_data()
    result_df = perform_ttest(df)
    top_genes = result_df['Gene'].head(50).tolist()
    X = df[top_genes]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return jsonify({'accuracy': acc})

@app.route('/api/covid/svm')
def svm():
    df = load_data()
    result_df = perform_ttest(df)
    top_genes = result_df['Gene'].head(500).tolist()
    X = df[top_genes]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    svm = GridSearchCV(SVC(), {'C':[0.1,1,10], 'kernel':['linear','rbf'], 'gamma':['scale','auto']}, cv=3)
    svm.fit(X_train, y_train)
    return jsonify({'best_accuracy': svm.best_score_})

@app.route('/api/covid/xgboost')
def xgboost():
    df = load_data()
    result_df = perform_ttest(df)
    top_genes = result_df['Gene'].head(50).tolist()
    X = df[top_genes]
    y = df['label'].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8)
    model.fit(X_train, y_train)
    return jsonify({'xgboost_accuracy': model.score(X_test, y_test)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
