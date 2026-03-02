import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "heart_cleaned.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ACCURACY_PATH = os.path.join(MODELS_DIR, "accuracy.pkl")

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ heart_cleaned.csv not found in: {BASE_DIR}")
        st.stop()
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    return df

# ============================================================================
# MODEL TRAINING
# ============================================================================
@st.cache_resource
def train_model():
    df = load_data()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(accuracy, ACCURACY_PATH)

    return model, scaler, accuracy, X.columns.tolist()

# ============================================================================
# LOAD OR TRAIN MODEL
# ============================================================================
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ACCURACY_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        accuracy = joblib.load(ACCURACY_PATH)
        df = load_data()
        feature_names = df.drop("target", axis=1).columns.tolist()
        return model, scaler, accuracy, feature_names
    else:
        return train_model()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict(model, scaler, input_data):
    arr = np.array(input_data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)[0]
    probability = model.predict_proba(arr_scaled)[0]
    return prediction, probability

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("❤️ Heart Disease Risk Prediction System")
    st.markdown("Machine Learning-based Cardiovascular Risk Assessment")

    model, scaler, accuracy, feature_names = get_model()

    with st.sidebar:
        st.header("Model Information")
        st.success(f"Test Accuracy: {accuracy*100:.2f}%")
        st.markdown("""
        - Algorithm: Random Forest
        - Trees: 200
        - Features: 13
        - Dataset: UCI Cleveland
        """)
        st.warning("⚠️ For educational purposes only.")

    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Insights", "About"])

    # ===================== PREDICTION TAB =====================
    with tab1:
        st.subheader("Enter Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 20, 100, 50)
            sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
            cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])

        with col2:
            trestbps = st.slider("Resting BP", 80, 200, 120)
            chol = st.slider("Cholesterol", 100, 400, 200)
            fbs = st.selectbox("Fasting Blood Sugar >120 (0/1)", [0, 1])

        with col3:
            restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
            thalach = st.slider("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Angina (0/1)", [0, 1])

        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope (0-2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

        if st.button("Predict Risk"):
            input_data = [
                age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak,
                slope, ca, thal
            ]

            prediction, probability = predict(model, scaler, input_data)

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("⚠️ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={"text": "Risk Percentage"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if probability[1] > 0.5 else "green"},
                }
            ))
            st.plotly_chart(fig, width="stretch")

            st.write(f"Probability (No Disease): {probability[0]*100:.2f}%")
            st.write(f"Probability (Disease): {probability[1]*100:.2f}%")

    # ===================== MODEL INSIGHTS =====================
    with tab2:
        st.subheader("Feature Importance")

        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig_imp = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance"
        )
        st.plotly_chart(fig_imp, width="stretch")

        st.subheader("Dataset Overview")

        df = load_data()
        fig_target = px.pie(df, names="target", hole=0.4)
        st.plotly_chart(fig_target, width="stretch")

    # ===================== ABOUT =====================
    with tab3:
        st.markdown("""
        ### About This Project
        Developed for **STW5000CEM – Introduction to Artificial Intelligence**

        - Uses Random Forest Classifier  
        - Trained on UCI Cleveland Heart Dataset  
        - Predicts heart disease risk from 13 clinical parameters  

        ⚠️ This application is for academic and educational purposes only.
        """)

    st.markdown("---")
    st.caption("© 2026 Heart Disease Prediction System")

if __name__ == "__main__":
    main()