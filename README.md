# Telco Customer Churn — Prédiction du churn

Pipeline de machine learning de bout en bout pour prédire le départ (*churn*) de clients
télécom, de l'analyse des données au déploiement d'une API. L'objectif est **métier** :
détecter un maximum de clients sur le point de partir pour pouvoir les retenir.

> Chaque décision méthodologique est justifiée dans [`cahier des charges.md`](cahier%20des%20charges.md).

---

## Contexte & données

- **Dataset** : [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7 043 clients, 21 colonnes.
- **Cible** : `Churn` (Yes/No), déséquilibrée à **26,5 % / 73,5 %**.
- **Features** : 19 variables après exclusion de `customerID` (4 numériques, 15 catégorielles)
  — données démographiques, services souscrits, type de contrat et facturation.

## Méthodologie

| Étape | Choix clés |
|---|---|
| **1. Nettoyage** | `TotalCharges` (texte → numérique) ; 11 valeurs vides = clients `tenure=0`, imputées à 0 (cause logique, pas approximation). |
| **2. EDA** | Catégorielles : Chi² + **V de Cramér** (taille d'effet, plus fiable que la p-value à n=7043). Numériques : distributions par classe, corrélations, outliers (IQR de Tukey). |
| **3. Preprocessing** | **Anti-fuite** : split d'abord, transformations apprenantes refit par fold. Deux pipelines : one-hot + StandardScaler (LR), encodage ordinal sans scaling (XGBoost). Aucune feature dérivée (choix assumé). |
| **4. Déséquilibre** | Pondération de classe (`class_weight='balanced'` / `scale_pos_weight`) + seuil ajusté. Pas de SMOTE (simplicité, pas de risque de fuite). |
| **5. Modélisation** | StratifiedKFold 5 plis. Baseline = régression logistique (sans tuning). Modèle principal = **XGBoost réglé par Optuna** (optimisation bayésienne, objectif PR-AUC). |
| **6. Évaluation** | Seuil de décision calibré pour **recall ≥ 0.80** (precision max sous contrainte), calé sur l'out-of-fold du train puis appliqué au test intact. |
| **7. Interprétabilité** | **SHAP** global (importance, beeswarm) + local (waterfalls), recoupé avec l'EDA. |
| **8. Déploiement** | Pipeline + seuil sérialisés (`joblib`), **API FastAPI** `/predict`, conteneurisation Docker. |

## Résultats

Performance finale sur le **test set** (1 409 clients, jamais vu pendant l'entraînement) :

| Métrique | LR (seuil 0.5) | XGBoost (seuil 0.5) | **XGBoost (seuil métier 0.51)** |
|---|---|---|---|
| Recall (churn) | 0.783 | 0.797 | **0.789** |
| Precision (churn) | 0.504 | 0.513 | **0.514** |
| F1 (churn) | 0.614 | 0.624 | **0.622** |
| ROC-AUC | 0.842 | 0.846 | **0.846** |
| PR-AUC | 0.633 | 0.663 | **0.663** |

Le XGBoost réglé domine la baseline sur toutes les métriques, en particulier la **PR-AUC**
(la plus pertinente en contexte déséquilibré). Au seuil métier, il capte **295 churners sur 374**
(recall 0.79) sur le test.

> ℹ️ Le seuil est calibré sur l'out-of-fold du train pour viser recall = 0.80 ; sur le test il
> atteint 0.79 — l'écart de généralisation attendu d'un seuil jamais ajusté sur le test (pas de fuite).

### Insights business (EDA + confirmés par SHAP)

1. **Le type de contrat domine** : les contrats *month-to-month* churnent massivement, les contrats 1–2 ans très peu. Levier de rétention n°1.
2. **Faible ancienneté + facture mensuelle élevée** = profil à risque le plus net (nouveaux clients chers, souvent en *fiber optic*).
3. **L'absence de sécurité/support en ligne** (`OnlineSecurity`, `TechSupport`) est associée au sur-churn → piste de bundles de fidélisation.
4. **`gender` et `PhoneService` n'ont aucun pouvoir prédictif** — confirmé par le Chi² **et** par SHAP (double validation).

SHAP corrige aussi une sur-estimation de l'EDA univariée : `DeviceProtection`/`OnlineBackup`,
forts isolément, sont quasi nuls une fois la **redondance** prise en compte par le modèle.

## Structure du projet

```
.
├── sample.ipynb              # Notebook complet (étapes 1 à 8)
├── cahier des charges.md     # Décisions et justifications
├── models/
│   └── churn_pipeline.joblib # Pipeline entraîné + seuil métier (sérialisé en §8)
├── app/
│   └── main.py               # API FastAPI (endpoint /predict)
├── Dockerfile                # Image de production
├── requirement.txt           # Dépendances du notebook (entraînement)
└── requirements-api.txt      # Dépendances de l'API (déploiement)
```

## Lancement

### 1. Notebook (entraînement)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows : .venv\Scripts\activate
pip install -r requirement.txt
jupyter notebook sample.ipynb     # nécessite une connexion (données chargées depuis une URL)
```

L'exécution complète régénère `models/churn_pipeline.joblib`.

### 2. API en local

```bash
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

Documentation interactive (Swagger UI) : <http://localhost:8000/docs>

Exemple de requête :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":1,"Partner":"No","Dependents":"No","tenure":2,
       "PhoneService":"Yes","MultipleLines":"Yes","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"No","StreamingMovies":"Yes","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":84.05,"TotalCharges":186.05}'
# -> {"churn_probability":0.9267,"churn":true,"threshold":0.5086}
```

### 3. API via Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Stack technique

Python 3.13 · pandas · scikit-learn · XGBoost · Optuna · SHAP · FastAPI · Docker
