import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

# Configuration Streamlit
sns.set(style="whitegrid", palette="muted")
st.set_page_config(layout="wide")
st.title("📊 Projet de Segmentation de Clients")
st.subheader("Auteur: Diop Abdoulaye")
st.markdown("Source : Wholesale customers dataset")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")


data = load_data()
numeric_vars = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# Menu latéral
menu = st.sidebar.selectbox("Menu principal", ["📁 Affichage des données", "📊 Analyse exploratoire", "🧩 Segmentation"])

# ------------------------ Affichage des données ------------------------
if menu == "📁 Affichage des données":
    st.header("🔍 Aperçu des données")
    st.write(data.head())
    st.write("📏 Dimensions :", data.shape)
    st.write("📌 Statistiques descriptives :")
    st.write(data.describe())

# ------------------------ Analyse exploratoire ------------------------
elif menu == "📊 Analyse exploratoire":
    st.header("📈 Analyse exploratoire")

    st.subheader("Variables catégoriques")
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Afficher Channel"):
            fig, ax = plt.subplots()
            sns.countplot(x='Channel', data=data, ax=ax)
            ax.set_title("Répartition par canal de distribution (Channel)")
            st.pyplot(fig)
    with col2:
        if st.checkbox("Afficher Region"):
            fig, ax = plt.subplots()
            sns.countplot(x='Region', data=data, ax=ax)
            ax.set_title("Répartition par région (Region)")
            st.pyplot(fig)

    st.subheader("📌 Analyse des variables numériques")
    selected_vars = st.multiselect("Choisissez les variables numériques à explorer :", numeric_vars)

    for var in selected_vars:
        st.markdown(f"### 📍 {var}")
        st.write(data[var].describe())

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(data[var], kde=True, bins=30, ax=ax1)
            ax1.set_title(f"Distribution de {var}")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=data[var], ax=ax2)
            ax2.set_title(f"Boxplot de {var}")
            st.pyplot(fig2)

    if st.checkbox("Afficher la matrice de corrélation"):
        st.subheader("🔗 Matrice de corrélation")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[numeric_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# ------------------------ Segmentation ------------------------
elif menu == "🧩 Segmentation":
    st.header("🧠 Réduction de dimension et Clustering")

    X = data[numeric_vars]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Réduction de dimension
    method = st.selectbox("Choisissez la méthode de réduction :", ["PCA Linéaire", "Kernel PCA (Non Linéaire)"])

    if method == "PCA Linéaire":
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)
        corr = pearsonr(X_reduced[:, 0], X_reduced[:, 1])[0]
        reduction_type = "PCA linéaire"
    else:
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
        X_reduced = kpca.fit_transform(X_scaled)
        corr = pearsonr(X_reduced[:, 0], X_reduced[:, 1])[0]
        reduction_type = "Kernel PCA"

    st.markdown(f"📊 Corrélation entre PC1 et PC2 ({reduction_type}) : `{corr:.4f}`")

    fig1, ax1 = plt.subplots()
    ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c='skyblue', alpha=0.7)
    ax1.set_title(f"Visualisation 2D ({reduction_type})")
    ax1.set_xlabel("Composante 1")
    ax1.set_ylabel("Composante 2")
    st.pyplot(fig1)

    # Clustering
    k = st.slider("Choisissez le nombre de clusters :", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)

    st.success(f"Clustering effectué avec **k={k}**")

    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='Set1', s=60, alpha=0.8)
    ax2.set_title(f"Clusters K-means (k={k}) - {reduction_type}")
    ax2.set_xlabel("Composante 1")
    ax2.set_ylabel("Composante 2")
    st.pyplot(fig2)
