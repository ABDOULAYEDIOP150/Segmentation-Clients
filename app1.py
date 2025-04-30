import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

# -------------------- Configuration générale --------------------
sns.set(style="whitegrid", palette="muted")
st.set_page_config(layout="wide", page_title="Segmentation Clients", page_icon="📊")
st.title("📊 Projet de Segmentation de Clients")
st.markdown("**Auteur : Diop Abdoulaye**")
st.caption("Source : Wholesale customers dataset")

# -------------------- Objectif --------------------
st.markdown("""
### 🎯 Objectif du projet :
L'objectif de ce projet est de segmenter des clients d'un grossiste alimentaire en groupes homogènes 
à partir de leurs comportements d’achat. À la fin, on souhaite pouvoir :
- Visualiser ces groupes (ou "segments") sur une carte 2D ;
- Identifier à quel segment un nouvel utilisateur appartient selon ses dépenses ;
- Aider une entreprise à adapter ses offres ou stratégies selon chaque type de client.
""")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")


data = load_data()
numeric_vars = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# -------------------- Menu latéral --------------------
menu = st.sidebar.selectbox("📌 Menu principal", ["📁 Données", "📊 Analyse exploratoire", "🧩 Segmentation", "🔍 Prédiction"])

# -------------------- 1. Affichage des données --------------------
if menu == "📁 Données":
    st.header("📄 Aperçu du jeu de données")

    st.markdown("### 👀 Affichage interactif")
    n_rows = st.slider("Nombre de lignes à afficher :", min_value=5, max_value=len(data), value=5)
    st.dataframe(data.head(n_rows), use_container_width=True)
    st.markdown(f"**Dimensions :** {data.shape[0]} lignes × {data.shape[1]} colonnes")

    st.markdown("### 📝 Description des variables")
    st.markdown("""
    - **Channel** : Type de canal (1 = Horeca (Hotel/Restaurant/Café), 2 = Détaillant).
    - **Region** : Région du client (1 = Lisbonne, 2 = Sud, 3 = Région intérieure).
    - **Fresh** : Dépenses annuelles (en unité monétaire) en produits frais.
    - **Milk** : Dépenses annuelles en lait et produits laitiers.
    - **Grocery** : Dépenses annuelles en épicerie.
    - **Frozen** : Dépenses annuelles en produits surgelés.
    - **Detergents_Paper** : Dépenses annuelles en détergents et papier.
    - **Delicassen** : Dépenses annuelles en produits fins (traiteur, spécialités).
    """)

    st.markdown("**Statistiques descriptives :**")
    st.dataframe(data.describe(), use_container_width=True)

# -------------------- 2. Analyse exploratoire --------------------
elif menu == "📊 Analyse exploratoire":
    st.header("📈 Analyse exploratoire")
    st.markdown("Cette section permet d'explorer la distribution des variables.")

    st.subheader("📋 Variables catégorielles")
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Afficher la distribution de `Channel`"):
            fig, ax = plt.subplots()
            sns.countplot(x='Channel', data=data, ax=ax)
            ax.set_title("Répartition par canal de distribution")
            st.pyplot(fig)
    with col2:
        if st.checkbox("Afficher la distribution de `Region`"):
            fig, ax = plt.subplots()
            sns.countplot(x='Region', data=data, ax=ax)
            ax.set_title("Répartition par région")
            st.pyplot(fig)

    st.subheader("📊 Variables numériques")
    selected_vars = st.multiselect("Choisissez les variables à explorer :", numeric_vars)

    for var in selected_vars:
        st.markdown(f"#### 🔹 {var}")
        st.write(data[var].describe())

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            sns.histplot(data[var], kde=True, bins=30, ax=ax)
            ax.set_title(f"Histogramme de {var}")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots()
            sns.boxplot(x=data[var], ax=ax)
            ax.set_title(f"Boxplot de {var}")
            st.pyplot(fig)

    if st.checkbox("Afficher la matrice de corrélation"):
        st.subheader("🔗 Matrice de corrélation")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[numeric_vars].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# -------------------- 3. Segmentation --------------------
elif menu == "🧩 Segmentation":
    st.header("🧠 Réduction de dimension et Clustering")
    st.markdown("On privilégie la méthode présentant la plus faible corrélation entre les composantes, car plus celles-ci sont décorrélées, meilleure est la qualité de la réduction.")
    X = data[numeric_vars]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    method = st.selectbox("Méthode de réduction de dimension :", ["PCA Linéaire", "Kernel PCA (Non Linéaire)"])

    if method == "PCA Linéaire":
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)

    X_reduced = reducer.fit_transform(X_scaled)
    corr = pearsonr(X_reduced[:, 0], X_reduced[:, 1])[0]
    st.markdown("📉 Corrélation entre les composantes réduites :")
    st.latex(f"\\text{{Corrélation}} = {corr:.2e}")

    fig, ax = plt.subplots()
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, color='skyblue')
    ax.set_title("Visualisation 2D après réduction")
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    st.pyplot(fig)

    k = st.selectbox("Choisissez le nombre de clusters :", list(range(2, 11)))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_reduced)
    st.success(f"✅ Clustering effectué avec **k = {k}**")

    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='Set1', s=60, alpha=0.8)
    ax2.set_title("Résultat du K-means")
    ax2.set_xlabel("Composante 1")
    ax2.set_ylabel("Composante 2")
    st.pyplot(fig2)

    st.markdown("### 🧾 Légende des segments")

    for i in range(k):
        rgba = plt.cm.Set1(i / max(1, k - 1))
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <div style="width: 18px; height: 18px; background-color: {hex_color}; border-radius: 3px;"></div>
                <span style="color: {hex_color}; font-weight: bold;">Segment {i+1}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.session_state.update({
            'scaler': scaler,
            'reducer': reducer,
            'X_reduced': X_reduced,
            'clusters': clusters,
            'kmeans': kmeans
        })
        
        # Afficher 5 clients pour chaque segment
    st.markdown("### 👥 Exemples de clients par segment")
    data_with_clusters = data.copy()
    data_with_clusters['Segment'] = (clusters+1)

    for i in range(k):
        st.markdown(f"#### 🔸 Segment {i+1}")
        segment_clients = data_with_clusters[data_with_clusters['Segment'] == (i+1)]
        # Affichage des 5 premiers clients avec les indices du dataframe
        segment_clients_display = segment_clients.head(5)
        segment_clients_display.index = [f"client_{idx+1}" for idx in segment_clients_display.index]
        st.dataframe(segment_clients_display, use_container_width=True)
        
    st.markdown("### 🧠 Interprétation des segments")

    st.markdown("""
        - Après avoir appliqué une méthode de segmentation **(par clustering)** sur notre base de clients, nous avons pu identifier plusieurs groupes distincts partageant des caractéristiques communes. Ces segments ont été formés à partir de données telles que les **habitudes d'achat**, la **fréquence des visites** ou encore les **montants dépensés**.

        - Afin de donner du sens à ces segments au-delà des seuls critères statistiques, **un travail collaboratif sera mené avec l'équipe marketing**. Ensemble, nous analyserons  le **profil type** de chaque segment afin de mieux comprendre leurs spécificités. Ce travail d’interprétation permettra d’associer à chaque groupe un type de segmentation pertinent — par exemple **comportementale**, **sociodémographique** ou **liée à la valeur client**.

        - Cette démarche  facilitera la construction de **profils clients concrets et exploitables**, en vue d’adapter les **stratégies commerciales**, les **campagnes de communication**, ou encore les **offres proposées**.
        """)

# -------------------- 4. Prédiction --------------------
elif menu == "🔍 Prédiction":
    st.header("🔮 Prédire le segment d’un utilisateur")

    with st.form("form_user"):
        st.write("**Données de consommation :**")
        user_input = {var: st.number_input(f"{var}", min_value=0.0, step=1.0) for var in numeric_vars}
        submitted = st.form_submit_button("Prédire")

    if submitted:
        required_keys = ['scaler', 'reducer', 'kmeans']
        if all(key in st.session_state for key in required_keys):
            user_df = pd.DataFrame([user_input])
            user_scaled = st.session_state['scaler'].transform(user_df)
            user_reduced = st.session_state['reducer'].transform(user_scaled)
            prediction = st.session_state['kmeans'].predict(user_reduced)[0]

            st.success(f"🧾 L’utilisateur est classé dans le **segment {prediction}**")

            fig3, ax3 = plt.subplots()
            ax3.scatter(st.session_state['X_reduced'][:, 0], st.session_state['X_reduced'][:, 1],
                        c=st.session_state['clusters'], cmap='Set1', alpha=0.5, s=60, label="Clients")
            ax3.scatter(user_reduced[:, 0], user_reduced[:, 1], color='black', s=100, marker='X', label='Utilisateur')
            ax3.set_title("Position de l'utilisateur")
            ax3.set_xlabel("Composante 1")
            ax3.set_ylabel("Composante 2")
            ax3.legend()
            st.pyplot(fig3)
        else:
            st.error("⚠️ Veuillez d'abord effectuer une segmentation dans l’onglet '🧩 Segmentation'.")
