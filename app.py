import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional charting dependency used only for the model-information page.
import matplotlib.pyplot as plt


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_rf_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"

FEATURE_ORDER = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

CATEGORICAL_COLUMNS = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(0, 200, 255, 0.08), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(80, 120, 255, 0.07), transparent 25%),
            #07111f;
        color: #e8f1ff;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.72);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081423 0%, #0b1727 100%);
        border-right: 1px solid rgba(110, 190, 255, 0.12);
    }

    .hero {
        padding: 28px 32px;
        border: 1px solid rgba(80, 210, 255, 0.18);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(10, 33, 56, .92), rgba(8, 19, 34, .88));
        box-shadow: 0 18px 55px rgba(0,0,0,.24);
        margin-bottom: 24px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -0.03em;
    }

    .hero p {
        color: #a9bdd6;
        margin: 8px 0 16px;
        font-size: 1.05rem;
    }

    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid rgba(75, 211, 255, .28);
        background: rgba(20, 170, 220, .08);
        color: #7ee8ff;
        font-size: .82rem;
        font-weight: 600;
    }

    .card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(11, 28, 47, .78);
        border: 1px solid rgba(140, 190, 230, .12);
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
    }

    .result-card {
        text-align: center;
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(7, 62, 84, .9), rgba(8, 29, 49, .95));
        border: 1px solid rgba(83, 225, 255, .25);
        box-shadow: 0 18px 50px rgba(0, 160, 220, .10);
        margin: 18px 0;
    }

    .result-label {
        color: #9bb5cc;
        font-size: .82rem;
        font-weight: 700;
        letter-spacing: .13em;
    }

    .result-number {
        color: #7ee8ff;
        font-size: 4rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 8px 0;
    }

    .result-sub {
        color: #b5c8db;
    }

    .metric-card {
        padding: 16px;
        border-radius: 16px;
        background: rgba(11, 28, 47, .72);
        border: 1px solid rgba(140, 190, 230, .11);
    }

    .metric-title {
        color: #8fa9c2;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .metric-value {
        color: #edf7ff;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 4px;
    }

    .section-title {
        margin: 12px 0 8px;
        font-size: 1.35rem;
        font-weight: 750;
    }

    .muted {
        color: #9eb1c7;
    }

    .footer {
        text-align: center;
        color: #72869d;
        padding: 28px 0 10px;
        font-size: .82rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 700;
        border: 1px solid rgba(90, 220, 255, .24);
    }

    div[data-testid="stMetric"] {
        background: rgba(11, 28, 47, .72);
        border: 1px solid rgba(140, 190, 230, .11);
        padding: 14px;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH.name}")
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_encoders():
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing encoder file: {ENCODER_PATH.name}")
    with ENCODER_PATH.open("rb") as f:
        return pickle.load(f)


def encode_inputs(values: dict, encoders: dict) -> pd.DataFrame:
    """Encode categorical inputs using the exact saved LabelEncoder objects."""
    row = {
        "Distance_km": float(values["Distance_km"]),
        "Weather": encoders["Weather"].transform([values["Weather"]])[0],
        "Traffic_Level": encoders["Traffic_Level"].transform([values["Traffic_Level"]])[0],
        "Time_of_Day": encoders["Time_of_Day"].transform([values["Time_of_Day"]])[0],
        "Vehicle_Type": encoders["Vehicle_Type"].transform([values["Vehicle_Type"]])[0],
        "Preparation_Time_min": float(values["Preparation_Time_min"]),
        "Courier_Experience_yrs": float(values["Courier_Experience_yrs"]),
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def make_prediction(model, input_data: pd.DataFrame) -> float:
    prediction = model.predict(input_data)
    return float(np.asarray(prediction).ravel()[0])


def delivery_class(minutes: float):
    if minutes < 30:
        return "Fast Delivery", "Estimated delivery is under 30 minutes."
    if minutes <= 60:
        return "Standard Delivery", "Estimated delivery is within the 30–60 minute range."
    return "Longer Delivery", "Estimated delivery is above 60 minutes."


def metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Load artifacts
# -----------------------------
try:
    model = load_model()
    encoders = load_encoders()
    artifacts_ok = True
except Exception as exc:
    artifacts_ok = False
    model = None
    encoders = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🚴 Delivery AI")
    st.caption("Food Delivery Time Prediction")

    page = st.radio(
        "Navigation",
        ["Prediction", "Model Information", "About Project"],
        index=0,
    )

    st.divider()

    if artifacts_ok:
        st.success("Model Loaded ✓")
    else:
        st.error("Model Unavailable")

    st.markdown("**Model**  \nRandom Forest Regressor")
    st.markdown("**Task**  \nRegression")
    st.markdown("**Target**  \nDelivery Time")
    st.markdown("**Framework**  \nScikit-learn + Streamlit")


# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Food Delivery Time Predictor</h1>
        <p>AI-powered delivery time estimation using Random Forest Regression</p>
        <span class="badge">Machine Learning • Random Forest • Regression</span>
    </div>
    """,
    unsafe_allow_html=True,
)


if not artifacts_ok:
    st.error(
        "Unable to load the trained model or encoders. "
        "Please make sure `best_rf_model.pkl` and `label_encoders.pkl` "
        "are present beside `app.py`."
    )
    st.stop()


# -----------------------------
# Prediction page
# -----------------------------
if page == "Prediction":
    st.markdown('<div class="section-title">Delivery Information</div>', unsafe_allow_html=True)
    st.caption("Enter the delivery conditions below to estimate the expected delivery time.")

    with st.form("prediction_form"):
        left, right = st.columns(2, gap="large")

        with left:
            distance = st.number_input(
                "Delivery Distance (km)",
                min_value=0.59,
                max_value=19.99,
                value=10.0,
                step=0.1,
                help="Training data range: approximately 0.59–19.99 km.",
            )

            weather = st.selectbox(
                "Weather",
                ["Clear", "Foggy", "Rainy", "Snowy", "Windy"],
            )

            traffic = st.selectbox(
                "Traffic Level",
                ["Low", "Medium", "High"],
            )

            time_of_day = st.selectbox(
                "Time of Day",
                ["Morning", "Afternoon", "Evening", "Night"],
            )

        with right:
            vehicle = st.selectbox(
                "Vehicle Type",
                ["Bike", "Car", "Scooter"],
            )

            preparation = st.number_input(
                "Preparation Time (minutes)",
                min_value=5,
                max_value=29,
                value=17,
                step=1,
                help="Training data range: approximately 5–29 minutes.",
            )

            experience = st.number_input(
                "Courier Experience (years)",
                min_value=0,
                max_value=9,
                value=5,
                step=1,
                help="Training data range: 0–9 years.",
            )

        st.write("")
        submitted = st.form_submit_button("🚀 Predict Delivery Time", use_container_width=True)

    if submitted:
        values = {
            "Distance_km": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": preparation,
            "Courier_Experience_yrs": experience,
        }

        try:
            with st.spinner("Analyzing delivery conditions..."):
                input_data = encode_inputs(values, encoders)
                prediction = make_prediction(model, input_data)

            rounded_prediction = int(round(prediction))
            category, interpretation = delivery_class(prediction)

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">ESTIMATED DELIVERY TIME</div>
                    <div class="result-number">{rounded_prediction} min</div>
                    <div class="result-sub">Predicted by Random Forest Regression</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if category == "Fast Delivery":
                st.success(f"**{category}** — {interpretation}")
            elif category == "Standard Delivery":
                st.info(f"**{category}** — {interpretation}")
            else:
                st.warning(f"**{category}** — {interpretation}")

            st.markdown('<div class="section-title">Order Summary</div>', unsafe_allow_html=True)

            cols = st.columns(4)
            summary = [
                ("Distance", f"{distance:.1f} km"),
                ("Weather", weather),
                ("Traffic", traffic),
                ("Time of Day", time_of_day),
                ("Vehicle", vehicle),
                ("Preparation", f"{preparation} min"),
                ("Experience", f"{experience} yrs"),
                ("Prediction", f"{rounded_prediction} min"),
            ]

            for i, (title, value) in enumerate(summary):
                with cols[i % 4]:
                    metric_card(title, value)

        except ValueError as exc:
            st.error(
                "One of the selected categorical values is not supported by the "
                "saved encoder. Please verify the encoder file."
            )
        except Exception:
            st.error(
                "Prediction could not be completed. Please verify that the saved "
                "model, encoders, and feature schema are compatible."
            )


# -----------------------------
# Model Information page
# -----------------------------
elif page == "Model Information":
    st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        metric_card("Algorithm", "Random Forest Regressor")
    with b:
        metric_card("Problem Type", "Supervised Regression")
    with c:
        metric_card("Target", "Delivery Time (min)")

    st.write("")
    st.markdown(
        """
        <div class="card">
            <h3>How the model works</h3>
            <p class="muted">
            Random Forest Regression combines predictions from multiple decision
            trees and aggregates them to produce a robust numerical prediction.
            In this application, the trained model estimates delivery time from
            distance, weather, traffic, time of day, vehicle type, preparation
            time, and courier experience.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="section-title">Features Used</div>', unsafe_allow_html=True)

    feature_df = pd.DataFrame(
        {
            "Feature": FEATURE_ORDER,
            "Type": [
                "Numerical",
                "Categorical",
                "Categorical",
                "Categorical",
                "Categorical",
                "Numerical",
                "Numerical",
            ],
        }
    )
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "Feature": FEATURE_ORDER,
                "Importance": model.feature_importances_,
            }
        ).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importance")
        ax.grid(axis="x", alpha=0.18)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown('<div class="section-title">Actual Model Parameters</div>', unsafe_allow_html=True)
    params = model.get_params()
    params_df = pd.DataFrame(
        [{"Parameter": k, "Value": str(v)} for k, v in params.items()]
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)


# -----------------------------
# About page
# -----------------------------
else:
    st.markdown('<div class="section-title">About the Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <h2>Food Delivery Time Prediction</h2>
            <p class="muted">
            This machine learning project estimates food delivery duration from
            operational delivery conditions. The prediction considers delivery
            distance, weather, traffic level, time of day, vehicle type, food
            preparation time, and courier experience.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="section-title">ML Workflow</div>', unsafe_allow_html=True)

    workflow = [
        "Data Collection",
        "Data Cleaning",
        "Missing Value Handling",
        "Exploratory Data Analysis",
        "Categorical Encoding",
        "Train/Test Split",
        "Random Forest Regression",
        "Hyperparameter Tuning",
        "Model Evaluation",
        "Model Serialization",
        "Streamlit Deployment",
    ]

    for i, step in enumerate(workflow, start=1):
        st.markdown(f"**{i:02d}.** {step}")
        if i < len(workflow):
            st.caption("↓")

    st.write("")
    st.markdown('<div class="section-title">Dataset & Preprocessing</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - Original dataset: **1,000 records × 9 columns**
        - `Order_ID` was removed before modeling.
        - Missing values in `Weather`, `Traffic_Level`, and `Time_of_Day`
          were filled using the mode.
        - Missing `Courier_Experience_yrs` values were filled using the median.
        - Categorical features were encoded using saved `LabelEncoder` objects.
        - The saved Random Forest model is used directly for inference.
        """
    )

    st.write("")
    st.markdown(
        """
        <div class="footer">
            Built with Python • Scikit-learn • Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
