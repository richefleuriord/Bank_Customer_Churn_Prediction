# ============================================================
# Application Streamlit : Prédiction du Churn Client (Pipeline XGBoost)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# Config de la page
# ------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction du Churn Client",
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
# Chargement du pipeline XGBoost
# ------------------------------------------------------------
try:
    pipeline = joblib.load(r"C:\Users\HP\Course\phase_5\Models/Best_Model_Deployment.pkl")
    st.success("✅ Pipeline XGBoost chargé avec succès !")
except Exception as e:
    st.error(f"❌ Impossible de charger le pipeline : {e}")

# ------------------------------------------------------------
# Saisie utilisateur
# ------------------------------------------------------------
st.header("🧾 Informations du client")
with st.form("client_form"):
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        age = st.number_input("Âge", 18, 100, 30)
        tenure = st.number_input("Durée de relation (années)", 0, 20, 3)
        balance = st.number_input("Solde du compte", 0.0, 1_000_000.0, 5000.0, step=1000.0)
        num_products = st.selectbox("Nombre de produits détenus", [1, 2, 3])
        has_card = st.radio("Possède une carte de crédit ?", ["Oui", "Non"])
        is_active = st.radio("Client actif ?", ["Oui", "Non"])
    with col2:
        estimated_salary = st.number_input("Salaire estimé", 0.0, 1_000_000.0, 40000.0, step=1000.0)
        satisfaction = st.slider("Satisfaction Score", 0, 10, 5)
        points = st.number_input("Points accumulés", 0, 5000, 300)
        active_credit = st.radio("Crédit actif ?", ["Oui", "Non"])
        geography = st.selectbox("Pays", ["France", "Germany", "Spain"])
        gender = st.selectbox("Sexe", ["Homme", "Femme"])
        card_type = st.selectbox("Type de carte", ["SILVER", "GOLD", "PLATINUM"])
        age_category = st.selectbox("Catégorie d'âge", ["Jeune", "Moyen", "Âgé"])
        product_category = st.selectbox("Catégorie de produits", ["1", "2", "3+"])

    submitted = st.form_submit_button("🔮 Prédire le Churn")

# ------------------------------------------------------------
# Préparation des données utilisateur
# ------------------------------------------------------------
def prepare_input():
    data = {
        'CreditScore': [credit_score],
        'Tenure': [tenure],
        'Balance': [balance],
        'HasCrCard': [1 if has_card == "Oui" else 0],
        'IsActiveMember': [1 if is_active == "Oui" else 0],
        'EstimatedSalary': [estimated_salary],
        'Satisfaction Score': [satisfaction],
        'Point Earned': [points],
        'BalanceToSalaryRatio': [balance / estimated_salary if estimated_salary > 0 else 0],
        'ActiveCredit': [1 if active_credit == "Oui" else 0],
        'LowSatisfaction': [1 if satisfaction < 5 else 0],
        'Geography_Germany': [1 if geography == "Germany" else 0],
        'Geography_Spain': [1 if geography == "Spain" else 0],
        'Gender_Male': [1 if gender == "Homme" else 0],
        'Card Type_GOLD': [1 if card_type == "GOLD" else 0],
        'Card Type_PLATINUM': [1 if card_type == "PLATINUM" else 0],
        'Card Type_SILVER': [1 if card_type == "SILVER" else 0],
        'AgeCategory_Moyen': [1 if age_category == "Moyen" else 0],
        'AgeCategory_Âgé': [1 if age_category == "Âgé" else 0],
        'ProductCategory_2': [1 if product_category == "2" else 0],
        'ProductCategory_3+': [1 if product_category == "3+" else 0]
    }
    return pd.DataFrame(data)

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
            st.success(f"✅ Le client est **susceptible de rester** ({(1-proba)*100:.2f}% de probabilité).")

        # Affichage professionnel avec metrics
        col1, col2 = st.columns(2)
        col1.metric("Probabilité de rester", f"{(1-proba)*100:.2f}%")
        col2.metric("Probabilité de quitter", f"{proba*100:.2f}%")

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

# ------------------------------------------------------------
# Pied de page
# ------------------------------------------------------------
st.markdown("---")
st.markdown("👨‍💻 **Réalisé par Riché Fleurinord et Micka Louis** — *Phase 5 : Projet Data Science & IA (Akademi)*")
st.caption("© 2025 - Application développée avec Streamlit et XGBoost")
