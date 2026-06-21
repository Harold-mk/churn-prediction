# Cahier des charges — Telco Customer Churn Prediction

**Repo :** `Harold-mk/churn-prediction`
**Dataset :** IBM Telco Customer Churn — 7043 lignes, 21 colonnes, déséquilibre 73,5% (No) / 26,5% (Yes)
**Statut au moment de la rédaction :** chargement des données effectué (1 cellule). Toutes les étapes ci-dessous restent à coder.

---

## Étape 1 — Nettoyage des données

- **`TotalCharges`** (actuellement `str`) : conversion en numérique (`pd.to_numeric(errors='coerce')`), puis imputation à **0** pour les lignes où `tenure = 0` (clients pas encore facturés — la valeur manquante a une cause logique identifiée, pas une approximation arbitraire).
- **`customerID`** : exclu des features (identifiant pur, aucune valeur prédictive).
- **Colonnes redondantes** (`OnlineSecurity`, `StreamingTV`, etc. avec modalité `"No internet/phone service"`) : **conservées telles quelles**, pas de fusion en `"No"`. Justification si demandée en entretien : XGBoost gère bien la redondance nativement, pas de perte d'info, simplification jugée non nécessaire pour les modèles utilisés.
- **Doublons** : vérification systématique (`df.duplicated().sum()`), traitement si présents.

## Étape 2 — EDA (analyse exploratoire)

- **Variables catégorielles** : analyse visuelle (taux de churn par modalité) **+ test du Chi²** pour vérifier la significativité statistique de l'association avec `Churn`.
- **Variables numériques** (`tenure`, `MonthlyCharges`, `TotalCharges`) : histogrammes + boxplots par classe Churn, matrice de corrélation, **détection formalisée des outliers** (IQR ou z-score).
- **Multivariée** : pas d'analyse croisée à ce stade — reportée naturellement à l'interprétabilité SHAP (étape 7).
- Objectif : sortir 3-4 insights business concrets pour nourrir le README et la présentation.

## Étape 3 — Feature engineering & preprocessing

- **Encodage** : **deux pipelines distincts** selon le modèle :
  - Régression logistique → one-hot encoding + **StandardScaler** sur les variables numériques.
  - XGBoost → encodage natif/catégoriel, **pas de scaling** (non nécessaire pour les modèles à base d'arbres).
- **Features dérivées** : **aucune**. Choix assumé : pas de feature engineering manuel, on s'appuie sur l'encodage et la capacité du modèle (notamment XGBoost) à capturer les interactions. À assumer en entretien si la question du feature engineering est posée directement.
- **Split train/test** : 80/20, **stratifié sur `Churn`** (préserve le ratio 73,5/26,5 dans les deux sets).

## Étape 4 — Gestion du déséquilibre des classes

- **Pondération de classe** à l'entraînement (`class_weight='balanced'` pour la régression logistique, `scale_pos_weight` pour XGBoost).
- **Seuil de décision ajusté a posteriori** (voir étape 6).
- **Pas de SMOTE** : choix assumé pour éviter la complexité et le risque de fuite de données (notamment si mal appliqué avant le split). Pondération + seuil réfléchi jugés suffisants et plus simples à défendre.

## Étape 5 — Modélisation

- **Validation** : StratifiedKFold cross-validation (pas de split unique).
- **Baseline** : régression logistique avec `class_weight='balanced'`, **pas de tuning** — reste volontairement simple pour servir de point de comparaison honnête.
- **Modèle principal** : XGBoost, **tuning via Optuna** (optimisation bayésienne).
  - Hyperparamètres ciblés : `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`.
  - ⚠️ Point de vigilance : comprendre le fonctionnement d'Optuna (optimisation bayésienne) avant de l'utiliser en boîte noire — prévoir un détour théorique si besoin, sur le même principe que la régression logistique.

## Étape 6 — Évaluation

- **Métriques retenues** : ROC-AUC, PR-AUC, recall/precision/F1 sur la classe `Yes`, matrice de confusion.
- **Choix du seuil de décision** : objectif métier explicite — **recall ≥ 0.80** (capturer au moins 80% des vrais churners), seuil exact déterminé via la courbe precision-recall pour maximiser la precision sous cette contrainte. Plus défendable qu'un seuil par défaut à 0.5 ou qu'une simple maximisation automatique du F1.
- **Comparaison baseline vs XGBoost** : tableau comparatif complet sur toutes les métriques — pièce centrale du README/présentation.

## Étape 7 — Interprétabilité (SHAP)

- **Portée** : analyse **globale** (summary plot, importance des features) **+ locale** (2-3 prédictions individuelles expliquées via waterfall/force plot).
- **Recoupement avec l'EDA** : vérifier la cohérence entre les features importantes selon SHAP et les insights sortis à l'étape 2 (Chi², visualisations). Toute divergence est un point d'analyse à creuser et mentionner, pas à cacher.

## Étape 8 — Déploiement

- **API** : FastAPI, endpoint `/predict` qui prend les features d'un client en entrée et retourne la probabilité de churn + la classe prédite (selon le seuil fixé à l'étape 6).
- **Containerisation** : Docker (Dockerfile packageant l'API + le modèle sauvegardé en `pickle`/`joblib`).
- **Interface** : **aucune** pour l'instant — l'API seule + Swagger UI (auto-généré par FastAPI) suffisent comme preuve de compétence déploiement. Un ajout Streamlit reste possible plus tard en bonus, pas un prérequis.

## Étape 9 — Documentation

- README complet (contexte, données, méthodologie, résultats, instructions de lancement).
- Description GitHub (champ "About") — à rédiger une fois le projet réellement terminé, pas avant.

---

## Notes de méthode

- Chaque décision ci-dessus a été choisie avec une justification explicite, pas par défaut — c'est cette justification qu'il faut pouvoir restituer en entretien, pas seulement le résultat.
- Avancer étape par étape, dans l'ordre, une session de travail = une étape (ou sous-partie d'étape).
- Ne pas annoncer un statut d'avancement (ex: "complet avec déploiement") sans vérification de l'état réel du code.