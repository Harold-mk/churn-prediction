# API de prédiction du churn — image de production
FROM python:3.13-slim

WORKDIR /app

# Dépendances d'abord (cache Docker tant que requirements ne change pas)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Modèle sérialisé (§8 du notebook) + code de l'API
COPY models/churn_pipeline.joblib models/churn_pipeline.joblib
COPY app/ app/

EXPOSE 8000

# Doc interactive : http://localhost:8000/docs
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
