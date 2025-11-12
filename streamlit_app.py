# streamlit_app.py

import streamlit as st
import requests
import json

# --- Configuration de l'API ---
# Assurez-vous que l'API FastAPI est lancée sur ce port !
API_URL = "http://127.0.0.1:8000/predict"

# --- Configuration de la Page Streamlit ---
st.set_page_config(page_title="Prédicteur de Prix de Logements en Californie", layout="wide")

st.title("🏡 Modèle de Prédiction de Prix de Logements (California Housing)")
st.markdown("---")

st.header("Entrez les Caractéristiques du Quartier")

# --- Formulaire de Saisie des Données ---
with st.form(key='housing_form'):
    # Utilisation de colonnes pour une meilleure mise en page
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Localisation")
        # Longitude et Latitude (utilisées pour le clustering)
        longitude = st.number_input("Longitude (ex: -118.45 pour LA)", value=-118.45, step=0.01, format="%.2f")
        latitude = st.number_input("Latitude (ex: 34.00 pour LA)", value=34.00, step=0.01, format="%.2f")
        
        st.subheader("Caractéristiques du Logement")
        house_age = st.number_input("Âge médian des Maisons (années)", value=25.0, min_value=1.0, max_value=52.0)
        ave_rooms = st.number_input("Moyenne des Pièces par logement", value=5.5, min_value=1.0, max_value=20.0, format="%.2f")
        ave_bedrms = st.number_input("Moyenne des Chambres par logement", value=1.0, min_value=0.5, max_value=5.0, format="%.2f")
        
    with col2:
        st.subheader("Démographie & Économie")
        # MedInc est la feature clé
        med_inc = st.number_input("Revenu Médian (MedInc - en 100k USD)", value=4.5, min_value=0.5, max_value=15.0, format="%.2f", help="Feature clé : 4.5 équivaut à 45 000$")
        population = st.number_input("Population du Bloc", value=1500.0, min_value=10.0, max_value=10000.0)
        ave_occup = st.number_input("Moyenne d'Occupants par Logement", value=2.5, min_value=1.0, max_value=5.0, format="%.2f")
    
    # Bouton de soumission
    st.markdown("---")
    submitted = st.form_submit_button("Calculer le Prix Estimé 🚀")

# --- Logique de Prédiction ---
if submitted:
    # 1. Préparer les données au format JSON attendu par FastAPI
    input_data = {
        "Longitude": longitude,
        "Latitude": latitude,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "MedInc": med_inc
    }
    
    # 2. Appel à l'API FastAPI
    try:
        response = requests.post(API_URL, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Affichage des résultats
            st.success("✅ Prédiction Réussie !")
            
            st.metric(
                label="Prix Médian Estimé (USD)", 
                value=result['predicted_price_USD'],
                delta=f"Cluster Géographique ID: {result['input_cluster']}"
            )
            
            # Afficher les features clés utilisées
            st.markdown(f"""
            <div style='background-color: #f0f0f5; padding: 10px; border-radius: 5px;'>
            **Détails de l'Analyse :**
            - **Revenu (MedInc)** : {med_inc*100000:,.0f} $
            - **Prix en 100k** : {result['predicted_price_100k']}
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error(f"Erreur de l'API : Statut {response.status_code}")
            st.json(response.json())
            
    except requests.exceptions.ConnectionError:
        st.error("❌ ERREUR : Connexion à l'API FastAPI impossible.")
        st.warning("Veuillez vous assurer que votre serveur FastAPI (`uvicorn main:app --reload`) est en cours d'exécution.")