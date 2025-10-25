# ============================================================
# Application Streamlit : Prédiction du Churn Client (Pipeline XGBoost)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# Config de la page
# ------------------------------------------------------------
st.set_page_config(
    page_title="🏦 Prédiction du Churn Client",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Titre et description
# ------------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🏦 Prédiction du Churn Client</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #34495E;'>Remplissez les informations du client pour prédire la probabilité de départ.</p>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Chargement du pipeline et des colonnes finales
# ------------------------------------------------------------
MODELS_PATH = os.path.join(os.getcwd(), "Models")
pipeline_path = os.path.join(MODELS_PATH, "Best_Model_Deployment.pkl")
columns_path = os.path.join(MODELS_PATH, "columns_final.pkl")

if os.path.exists(pipeline_path) and os.path.exists(columns_path):
    pipeline = joblib.load(pipeline_path)
    columns_final = joblib.load(columns_path)
    st.success("✔ Pipeline XGBoost chargé avec succès !")
else:
    st.error("❌ Pipeline ou colonnes finales introuvables dans le dossier Models.")
    st.stop()

# ------------------------------------------------------------
# Saisie utilisateur
# ------------------------------------------------------------
st.header("🧑‍💼 Informations du client")
with st.form("client_form"):
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        tenure = st.number_input("Durée de relation (années)", 0, 20, 3)
        balance = st.number_input("Solde du compte", 0.0, 1_000_000.0, 5000.0, step=1000.0)
        num_products = st.selectbox("Nombre de produits détenus", [1, 2, 3])
        has_card = st.radio("Possède une carte de crédit ?", ["Oui", "Non"])
        is_active = st.radio("Client actif ?", ["Oui", "Non"])
    with col2:
        estimated_salary = st.number_input("Salaire estimé", 0.0, 1_000_000.0, 40000.0, step=1000.0)
        satisfaction = st.slider("Satisfaction Score", 0, 10, 5)
        points = st.number_input("Points accumulés", 0, 5000, 300)
        geography = st.selectbox("Pays", ["France", "Germany", "Spain"])
        gender = st.selectbox("Sexe", ["Homme", "Femme"])
        card_type = st.selectbox("Type de carte", ["SILVER", "GOLD", "PLATINUM"])
        age_category = st.selectbox("Catégorie d'âge", ["Jeune", "Moyen", "Âgé"])

    submitted = st.form_submit_button("🔍 Prédire le Churn")

# ------------------------------------------------------------
# Préparation des données utilisateur
# ------------------------------------------------------------
def prepare_input():
    data = {
        'CreditScore': [credit_score],
        'Tenure': [tenure],
        'Balance': [balance],
        'EstimatedSalary': [estimated_salary],
        'Satisfaction Score': [satisfaction],
        'Point Earned': [points],
        'NumOfProductsBinary': [0 if num_products == 1 else 1],
        'HasCrCard': [1 if has_card == "Oui" else 0],
        'IsActiveMember': [1 if is_active == "Oui" else 0],
        'BalanceToSalaryRatio': [balance / (estimated_salary+1)],
        'ActiveCredit': [1 if (is_active=="Oui" and has_card=="Oui") else 0],
        'LowSatisfaction': [1 if satisfaction <= 2 else 0],
        # One-hot encoding basé sur votre préparation
        'Geography_Germany': [1 if geography=="Germany" else 0],
        'Geography_Spain': [1 if geography=="Spain" else 0],
        'Gender_Male': [1 if gender=="Homme" else 0],
        'Card Type_GOLD': [1 if card_type=="GOLD" else 0],
        'Card Type_PLATINUM': [1 if card_type=="PLATINUM" else 0],
        'Card Type_SILVER': [1 if card_type=="SILVER" else 0],
        'AgeCategory_Moyen': [1 if age_category=="Moyen" else 0],
        'AgeCategory_Âgé': [1 if age_category=="Âgé" else 0],
        'AgeCategory_Jeune': [1 if age_category=="Jeune" else 0]
    }

    input_df = pd.DataFrame(data)

    # Alignement avec les colonnes finales du train set
    for col in columns_final:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns_final]  # réordonne
    return input_df

# ------------------------------------------------------------
# Prédiction
# ------------------------------------------------------------
if submitted:
    try:
        input_df = prepare_input()
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        st.subheader("📊 Résultat de la prédiction")
        if prediction == 1:
            st.warning(f"⚠️ Le client risque de **quitter la banque** ({proba*100:.2f}% de probabilité).")
        else:
            st.success(f"✔ Le client est **susceptible de rester** ({(1-proba)*100:.2f}% de probabilité).")

        col1, col2 = st.columns(2)
        col1.metric("Probabilité de rester", f"{(1-proba)*100:.2f}%")
        col2.metric("Probabilité de quitter", f"{proba*100:.2f}%")

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

# ------------------------------------------------------------
# Pied de page
# ------------------------------------------------------------
st.markdown("---")
st.markdown("👨‍💼‍🧑 **Réalisé par Riché Fleurinord et Micka Louis** — *Phase 5 : Projet Data Science & IA (Akademi)*")
st.caption("© 2025 - Application développée avec Streamlit et XGBoost")
