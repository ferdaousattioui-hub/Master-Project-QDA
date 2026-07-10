import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap  

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Projet ML - LDA vs QDA Iris", layout="wide")

# --- BARRE LATÉRALE DE NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez la section :", ["1. Présentation (PDF)", "2. Dashboard Interactive LDA / QDA"])

# --- CONFIGURATION ESTHÉTIQUE ---
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'

# --- CHARGEMENT DU DATASET IRIS ---
@st.cache_data
def load_iris_data():
    iris = load_iris()
    return iris

# --- SECTION 1 : PRÉSENTATION PPT ---
if page == "1. Présentation (PDF)":
    st.title("📂 Présentation du Projet - LDA & QDA")
    st.write("Analyse Discriminante Linéaire et Quadratique.")
    
    file_id = "1Z851NjEXS3l4_GaaeRDWUW8sau_Lc_In"  
    lien_drive_embed = f"https://drive.google.com/file/d/{file_id}/preview"
    
    st.components.v1.html(
        f'<iframe src="{lien_drive_embed}" style="width:100%; height:750px;" frameborder="0" allowfullscreen></iframe>',
        height=750
    )

# --- SECTION 2 : DASHBOARD INTERACTIVE LDA / QDA ---
elif page == "2. Dashboard Interactive LDA / QDA":
    st.title("💻 Analyse Interactive : LDA vs QDA (Iris)")
    st.write("Comparez en direct la flexibilité géométrique entre une frontière linéaire (LDA) et une frontière quadratique (QDA).")

    # --- BARRE LATÉRALE DES PARAMÈTRES (INTERACTIF) ---
    st.sidebar.subheader("🎛️ Paramètres du Dataset")
    test_size_ratio = st.sidebar.slider("Proportion du jeu de test (%)", min_value=10, max_value=50, value=30, step=5) / 100.0
    
    # Choix des variables à afficher en direct
    iris = load_iris_data()
    feature_names_list = iris.feature_names
    
    st.sidebar.subheader("🌱 Sélection des Caractéristiques (2D)")
    feat1 = st.sidebar.selectbox("Axe X", feature_names_list, index=0)
    feat2 = st.sidebar.selectbox("Axe Y", feature_names_list, index=1)

    # Préparation des données dynamique
    X_full = pd.DataFrame(iris.data, columns=iris.feature_names)
    X_subset = X_full[[feat1, feat2]].values
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=test_size_ratio, random_state=42, stratify=y
    )

    # Standardisation kima f script dyalk exact
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraînement des deux modèles en parallèle
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    y_pred_lda = lda.predict(X_test_scaled)
    acc_lda = accuracy_score(y_test, y_pred_lda) * 100

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)
    y_pred_qda = qda.predict(X_test_scaled)
    acc_qda = accuracy_score(y_test, y_pred_qda) * 100

    # --- PANNEAU DES MÉTRIQUES ---
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="🎯 Précision LDA (Linéaire)", value=f"{acc_lda:.2f} %")
    with col_m2:
        st.metric(label="🎯 Précision QDA (Quadratique)", value=f"{acc_qda:.2f} %")

    st.markdown("---")

    # --- VISUALISATION DES FRONTIÈRES ---
    st.subheader("🗺️ Comparaison Cartographique des Frontières de Décision")
    
    with plt.style.context('default'):
        fig_boundaries, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor='white')
        colors_list = ["#4c72b0", "#dd8452", "#55a868"]
        cmap_fixed = ListedColormap(colors_list)

        # 1. Frontière LDA
        DecisionBoundaryDisplay.from_estimator(
            lda, X_train_scaled, response_method="predict", cmap=cmap_fixed, alpha=0.2, ax=ax_b1
        )
        scatter_lda = ax_b1.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=cmap_fixed, edgecolor="black", s=50)
        ax_b1.set_title("LDA : Séparation Rigide (Lignes Droites)", fontsize=12, fontweight='bold')
        ax_b1.set_xlabel(feat1)
        ax_b1.set_ylabel(feat2)

        # 2. Frontière QDA
        DecisionBoundaryDisplay.from_estimator(
            qda, X_train_scaled, response_method="predict", cmap=cmap_fixed, alpha=0.2, ax=ax_b2
        )
        ax_b2.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=cmap_fixed, edgecolor="black", s=50)
        ax_b2.set_title("QDA : Séparation Souple (Courbes Paraboliques)", fontsize=12, fontweight='bold')
        ax_b2.set_xlabel(feat1)

        plt.tight_layout()
        st.pyplot(fig_boundaries)

    st.markdown("---")

    # --- MATRICES DE CONFUSION EN PARALLÈLE ---
    st.subheader("📊 Analyse Parallèle des Erreurs (Matrices de Confusion)")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.write("**Matrice LDA**")
        with plt.style.context('default'):
            fig_cm_lda, ax_cm_lda = plt.subplots(figsize=(5, 4), facecolor='white')
            cm_lda = confusion_matrix(y_test, y_pred_lda)
            sns.heatmap(cm_lda, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm_lda, cbar=False)
            st.pyplot(fig_cm_lda)

    with col_c2:
        st.write("**Matrice QDA**")
        with plt.style.context('default'):
            fig_cm_qda, ax_cm_qda = plt.subplots(figsize=(5, 4), facecolor='white')
            cm_qda = confusion_matrix(y_test, y_pred_qda)
            sns.heatmap(cm_qda, annot=True, fmt='d', cmap='Oranges', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm_qda, cbar=False)
            st.pyplot(fig_cm_qda)
