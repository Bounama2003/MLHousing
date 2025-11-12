# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

# --- 1. Définition et Chargement des Modèles ---
MODEL_PATH = "random_forest_housing_pipeline.joblib"
KMEANS_PATH = "kmeans_geo_model.joblib"

try:
    # Le pipeline complet (Pre-traitement + RF)
    model = joblib.load(MODEL_PATH)
    # Le modèle K-Means pour la feature Geo_Cluster
    kmeans_model = joblib.load(KMEANS_PATH) 
    print("Modèles chargés avec succès.")
except FileNotFoundError as e:
    print(f"ERREUR: Fichier de modèle non trouvé : {e}. Le chargement de l'API est interrompu.")
    model = None
    kmeans_model = None

# --- 2. Définition du Schéma d'Entrée (Pydantic) ---
# L'API doit accepter les 8 features ORIGINALES, non transformées
class HousingData(BaseModel):
    Longitude: float
    Latitude: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    MedInc: float 
    
# --- 3. Initialisation de l'API ---
app = FastAPI(
    title="API de Prédiction de Prix de Logements",
    version="1.0"
)

# --- 4. Endpoint de Prédiction ---
@app.post("/predict")
def predict_price(data: HousingData) -> Dict[str, Any]:
    if model is None or kmeans_model is None:
        return {"error": "Modèle non chargé."}

    # Convertir les données Pydantic en DataFrame pour le Feature Engineering
    df_raw = pd.DataFrame([data.model_dump()])

    # --- Feature Engineering Interne (REPRODUCTION des étapes du notebook) ---
    
    # 1. Création des ratios (Étape 12)
    df_raw['Bedrooms_per_Room'] = df_raw['AveBedrms'] / df_raw['AveRooms']
    
    # 2. Transformation Log de MedInc (Étape 13)
    df_raw['MedInc_log'] = np.log1p(df_raw['MedInc'])
    
    # 3. Clustering Géographique (Étape 16)
    # NOTE : En théorie, les coordonnées devraient être standardisées. 
    # Pour cette démo, nous utilisons les coordonnées brutes pour la prédiction
    # du cluster, car le StandardScaler n'a pas été sauvegardé.
    geo_data = df_raw[['Latitude', 'Longitude']]
    df_raw['Geo_Cluster'] = kmeans_model.predict(geo_data)

    # 4. Sélectionner les 10 features FINALES dans l'ordre attendu par le pipeline
    final_features_order = ['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                            'AveOccup', 'Latitude', 'Longitude', 
                            'Bedrooms_per_Room', 'MedInc_log', 'Geo_Cluster']
    
    X_final = df_raw[final_features_order]
    
    # 5. Prédiction finale par le pipeline
    prediction_100k = model.predict(X_final)[0]
    
    # Convertir le résultat en USD pour la clarté
    prediction_usd = prediction_100k * 100000

    return {
        "predicted_price_100k": round(prediction_100k, 4),
        "predicted_price_USD": f"${prediction_usd:,.2f}",
        "input_cluster": int(df_raw['Geo_Cluster'].iloc[0])
    }