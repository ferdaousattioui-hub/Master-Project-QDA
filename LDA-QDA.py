#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pip
pip.main(['install', 'seaborn', 'matplotlib', 'pandas'])


# In[3]:


import numpy as np # <-- gère les calculs mathématiques et les tableaux,
import pandas as pd
import matplotlib.pyplot as plt # <-- crée la structure des graphiques
import seaborn as sns # <-- pour les cartes thermiques
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # <-- La fonction qui sépare nos données en deux groupes
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis# <-- nos algorithmes de classification
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap  

# 1. Chargement du dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Sélection de 2 caractéristiques pour la visualisation 2D
X_subset = X[['sepal length (cm)', 'sepal width (cm)']]
# On sélectionne uniquement deux variables pour pouvoir afficher le graphique en 2D.

# 2. Séparation Entraînement / Test (70% - 30%)
# On sépare les données en isolant 30% pour le test tout en conservant les proportions de chaque espèce.
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Standardisation
# On standardise les données pour que la différence d'échelle entre les variables ne biaise pas les calculs.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialisation et entraînement des modèles
# La LDA suppose une dispersion identique pour créer des frontières droites, tandis que la QDA adapte des frontières courbes.
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

# On entraîne les deux modèles sur les données d'apprentissage standardisées.
lda.fit(X_train_scaled, y_train)
qda.fit(X_train_scaled, y_train)

# 5. Prédictions
# On génère les prédictions des deux modèles sur les données de test que l'algorithme ne connaît pas.
y_pred_lda = lda.predict(X_test_scaled)
y_pred_qda = qda.predict(X_test_scaled)

# ==========================================
# 6. VISUALISATION DES MATRICES DE CONFUSION
# ==========================================
fig_cm, (ax_cm1, ax_cm2) = plt.subplots(1, 2, figsize=(12, 5))
#Le calcul et la création des graphiques
cm_lda = confusion_matrix(y_test, y_pred_lda)
cm_qda = confusion_matrix(y_test, y_pred_qda)

# Heatmap LDA
sns.heatmap(cm_lda, annot=True, fmt="d", cmap="Blues", ax=ax_cm1,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
ax_cm1.set_title(f"Matrice de Confusion : LDA\n(Accuracy: {accuracy_score(y_test, y_pred_lda):.1%})")
ax_cm1.set_ylabel("Vraie Classe")
ax_cm1.set_xlabel("Classe Prédite")
#Le dessin des cartes thermiques pour les 2.
# Heatmap QDA
sns.heatmap(cm_qda, annot=True, fmt="d", cmap="Greens", ax=ax_cm2,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
ax_cm2.set_title(f"Matrice de Confusion : QDA\n(Accuracy: {accuracy_score(y_test, y_pred_qda):.1%})")
ax_cm2.set_ylabel("Vraie Classe")
ax_cm2.set_xlabel("Classe Prédite")

plt.tight_layout()

# ==========================================
# 7. VISUALISATION DES FRONTIÈRES DE DÉCISION
# ==========================================
fig_boundaries, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# CORRECTION ICI : On transforme la liste de couleurs en une Colormap officielle Matplotlib
colors_list = ["#4c72b0", "#dd8452", "#55a868"]
cmap_fixed = ListedColormap(colors_list)

# Frontière LDA (droits)
DecisionBoundaryDisplay.from_estimator(
    lda, X_train_scaled, response_method="predict", cmap=cmap_fixed, alpha=0.3, ax=ax_b1
)
sns.scatterplot(
    x=X_test_scaled[:, 0], y=X_test_scaled[:, 1], hue=iris.target_names[y_test],
    palette=colors_list, edgecolor="black", linewidth=1, ax=ax_b1
)
ax_b1.set_title("Frontières de Décision : LDA (Linéaire)")
ax_b1.set_xlabel("Longueur Sépal (Standardisé)")
ax_b1.set_ylabel("Largeur Sépal (Standardisé)")

# Frontière QDA (courbes)
DecisionBoundaryDisplay.from_estimator(
    qda, X_train_scaled, response_method="predict", cmap=cmap_fixed, alpha=0.3, ax=ax_b2
)
sns.scatterplot(
    x=X_test_scaled[:, 0], y=X_test_scaled[:, 1], hue=iris.target_names[y_test],
    palette=colors_list, edgecolor="black", linewidth=1, ax=ax_b2, legend=False
)
ax_b2.set_title("Frontières de Décision : QDA (Quadratique)")
ax_b2.set_xlabel("Longueur Sépal (Standardisé)")

plt.tight_layout()
plt.show()

# Rapports textuels
print("\n=== RAPPORT TEXTUEL LDA ===")
print(classification_report(y_test, y_pred_lda, target_names=iris.target_names))

print("\n=== RAPPORT TEXTUEL QDA ===")
print(classification_report(y_test, y_pred_qda, target_names=iris.target_names))


# In[ ]:




