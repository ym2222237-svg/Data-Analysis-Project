import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Adult Census Income — XGBoost",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1f4e79;
        text-align: center;
        padding-bottom: 0.3rem;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f4e79 0%, #2e75b6 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1f4e79;
        border-left: 5px solid #2e75b6;
        padding-left: 0.7rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .prediction-box-pos {
        background: linear-gradient(135deg, #1a7a4a, #27ae60);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 16px rgba(39,174,96,0.35);
    }
    .prediction-box-neg {
        background: linear-gradient(135deg, #922b21, #e74c3c);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 16px rgba(231,76,60,0.35);
    }
    .stProgress > div > div { background-color: #2e75b6; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">💰 Adult Census Income Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">XGBoost Classification · EDA · Prediction · Model Evaluation</div>', unsafe_allow_html=True)
st.markdown("---")

# ──────────────────────────────────────────────
# Sidebar — navigation & upload
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/machine-learning.png", width=70)
    st.title("Navigation")
    page = st.radio("Go to", [
        "📤 Upload & Preview",
        "🔍 EDA",
        "🧠 Train Model",
        "🔮 Predict",
    ])
    st.markdown("---")
    st.info("**Workflow:**\n1. Upload `adult.csv`\n2. Explore EDA charts\n3. Train the model\n4. Predict new records")

# ──────────────────────────────────────────────
# Session-state helpers
# ──────────────────────────────────────────────
for key in ["df", "model", "accuracy", "report", "cm", "feature_names"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ──────────────────────────────────────────────
# Helper: load df
# ──────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    # clean column names
    df.columns = [c.strip().replace("-", ".") for c in df.columns]
    # replace '?' with NaN
    df.replace("?", np.nan, inplace=True)
    df.drop_duplicates(inplace=True)
    if "fnlwgt" in df.columns:
        df["fnlwgt"] = np.log1p(df["fnlwgt"])
    return df


# ══════════════════════════════════════════════
# PAGE 1 — Upload & Preview
# ══════════════════════════════════════════════
if page == "📤 Upload & Preview":
    st.markdown('<div class="section-header">Upload Dataset</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload the **adult.csv** file", type=["csv"])
    if uploaded:
        df = load_data(uploaded.read())
        st.session_state["df"] = df
        st.success(f"✅ Loaded **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")

        col1, col2, col3, col4 = st.columns(4)
        for col, label, value in zip(
            [col1, col2, col3, col4],
            ["Rows", "Columns", "Missing Values", "Duplicates"],
            [df.shape[0], df.shape[1], int(df.isnull().sum().sum()), 0],
        ):
            col.markdown(
                f'<div class="metric-card"><div class="metric-value">{value:,}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Data Types</div>', unsafe_allow_html=True)
            st.dataframe(df.dtypes.rename("dtype").reset_index().rename(columns={"index": "column"}), use_container_width=True)
        with col_b:
            st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
            miss = df.isnull().sum().reset_index()
            miss.columns = ["Column", "Missing"]
            miss = miss[miss["Missing"] > 0]
            if miss.empty:
                st.success("No missing values found.")
            else:
                st.dataframe(miss, use_container_width=True)

        st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    else:
        st.warning("⬆️ Please upload **adult.csv** to begin.")


# ══════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════
elif page == "🔍 EDA":
    df = st.session_state["df"]
    if df is None:
        st.warning("⬆️ Please upload data first (Page 1).")
    else:
        st.markdown('<div class="section-header">Income Distribution</div>', unsafe_allow_html=True)
        if "income" in df.columns:
            fig = px.pie(df, names="income", title="Income Distribution", hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Age Distribution by Income</div>', unsafe_allow_html=True)
            if "age" in df.columns and "income" in df.columns:
                fig = px.box(df, x="income", y="age", color="income",
                             title="Age Distribution by Income",
                             color_discrete_sequence=["#1f4e79", "#e74c3c"])
                fig.update_layout(title_x=0.5, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Hours per Week by Income</div>', unsafe_allow_html=True)
            if "hours.per.week" in df.columns and "income" in df.columns:
                fig = px.box(df, x="income", y="hours.per.week", color="income",
                             title="Working Hours by Income Level",
                             color_discrete_sequence=["#2e75b6", "#27ae60"])
                fig.update_layout(title_x=0.5, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Income by Education</div>', unsafe_allow_html=True)
        if "education" in df.columns and "income" in df.columns:
            fig = px.histogram(df, x="education", color="income", barmode="group",
                               title="Income Distribution by Education Level",
                               color_discrete_sequence=["#1f4e79", "#e74c3c"])
            fig.update_layout(title_x=0.5, xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Occupation vs Income</div>', unsafe_allow_html=True)
        if "occupation" in df.columns and "income" in df.columns:
            fig = px.histogram(df, y="occupation", color="income", barmode="group",
                               title="Occupation vs Income",
                               color_discrete_sequence=["#1f4e79", "#e74c3c"])
            fig.update_layout(height=600, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Correlation Heatmap (Numeric Features)</div>', unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            fig_corr, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
            st.pyplot(fig_corr)


# ══════════════════════════════════════════════
# PAGE 3 — Train Model
# ══════════════════════════════════════════════
elif page == "🧠 Train Model":
    df = st.session_state["df"]
    if df is None:
        st.warning("⬆️ Please upload data first (Page 1).")
    else:
        st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        n_estimators = col1.slider("n_estimators", 50, 500, 200, 50)
        max_depth = col2.slider("max_depth", 2, 12, 6)
        learning_rate = col3.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
        test_size = st.slider("Test split size", 0.1, 0.4, 0.2, 0.05)

        DROP_COLS = ["income", "native.country", "fnlwgt", "relationship",
                     "workclass", "race", "sex", "occupation"]
        available_drop = [c for c in DROP_COLS if c in df.columns and c != "income"]

        if st.button("🚀 Train XGBoost Model", type="primary", use_container_width=True):
            with st.spinner("Training in progress …"):
                df_model = df.copy()

                # encode target
                le = LabelEncoder()
                df_model["income"] = le.fit_transform(df_model["income"].astype(str))

                X = df_model.drop([c for c in DROP_COLS if c in df_model.columns], axis=1)
                y = df_model["income"]

                cat_cols = X.select_dtypes(include="object").columns.tolist()
                num_cols = X.select_dtypes(exclude="object").columns.tolist()

                num_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ])
                cat_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ])
                preprocessor = ColumnTransformer([
                    ("num", num_pipe, num_cols),
                    ("cat", cat_pipe, cat_cols),
                ])

                model = Pipeline([
                    ("preprocessing", preprocessor),
                    ("classifier", XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42,
                        eval_metric="logloss",
                    )),
                ])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                cm = confusion_matrix(y_test, y_pred)

                st.session_state["model"] = model
                st.session_state["accuracy"] = acc
                st.session_state["report"] = report
                st.session_state["cm"] = cm
                st.session_state["feature_names"] = X.columns.tolist()
                st.session_state["le"] = le

                joblib.dump(model, "/tmp/adult_xgboost.pkl")

            st.success(f"✅ Model trained! Accuracy: **{acc:.4f}**")

        if st.session_state["accuracy"] is not None:
            acc = st.session_state["accuracy"]
            report = st.session_state["report"]
            cm = st.session_state["cm"]

            st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            for col, label, value in zip(
                [c1, c2, c3, c4],
                ["Accuracy", "Precision (≤50K)", "Recall (≤50K)", "F1 (≤50K)"],
                [
                    f"{acc:.4f}",
                    f"{report.get('0', report.get('<=50K', {})).get('precision', 0):.4f}",
                    f"{report.get('0', report.get('<=50K', {})).get('recall', 0):.4f}",
                    f"{report.get('0', report.get('<=50K', {})).get('f1-score', 0):.4f}",
                ],
            ):
                col.markdown(
                    f'<div class="metric-card"><div class="metric-value">{value}</div>'
                    f'<div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["≤50K", ">50K"],
                        yticklabels=["≤50K", ">50K"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

            st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

            # Download model
            with open("adult_xgboost.pkl", "rb") as f:
                st.download_button(
                    "⬇️ Download Trained Model (.pkl)",
                    f,
                    file_name="adult_xgboost.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════
# PAGE 4 — Predict
# ══════════════════════════════════════════════
elif page == "🔮 Predict":
    model = st.session_state["model"]
    if model is None:
        st.warning("🧠 Please train the model first (Page 3).")
    else:
        st.markdown('<div class="section-header">Enter Individual Details</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=17, max_value=90, value=35)
            education_num = st.number_input("Education Num", min_value=1, max_value=16, value=10)
            capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)

        with col2:
            hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
            capital_loss = st.number_input("Capital Loss", min_value=0, max_value=4000, value=0)
            marital_status = st.selectbox("Marital Status", [
                "Never-married", "Married-civ-spouse", "Divorced",
                "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
            ])

        with col3:
            education = st.selectbox("Education", [
                "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
                "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
            ])

        st.markdown("---")

        if st.button("🔮 Predict Income", type="primary", use_container_width=True):
            input_dict = {
                "age": [age],
                "education.num": [education_num],
                "capital.gain": [capital_gain],
                "capital.loss": [capital_loss],
                "hours.per.week": [hours_per_week],
                "marital.status": [marital_status],
                "education": [education],
            }

            # Build input aligned with training features
            feature_names = st.session_state["feature_names"]
            input_df = pd.DataFrame(columns=feature_names)
            row = {}
            for feat in feature_names:
                if feat in input_dict:
                    row[feat] = input_dict[feat][0]
                else:
                    # fill numeric with median-ish, object with mode placeholder
                    row[feat] = 0
            input_df = pd.DataFrame([row])

            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            label = ">50K" if prediction == 1 else "≤50K"
            confidence = proba[prediction] * 100

            st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
            box_class = "prediction-box-pos" if prediction == 1 else "prediction-box-neg"
            emoji = "🟢" if prediction == 1 else "🔴"
            st.markdown(
                f'<div class="{box_class}">{emoji} Predicted Income: <strong>{label}</strong><br>'
                f'<span style="font-size:1rem;opacity:0.9">Confidence: {confidence:.1f}%</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### Probability Breakdown")
            prob_df = pd.DataFrame({
                "Income Class": ["≤50K", ">50K"],
                "Probability": [proba[0], proba[1]],
            })
            fig = px.bar(prob_df, x="Income Class", y="Probability", color="Income Class",
                         color_discrete_sequence=["#1f4e79", "#e74c3c"],
                         title="Class Probabilities", text_auto=".2%")
            fig.update_layout(yaxis_range=[0, 1], showlegend=False, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Batch Prediction (CSV Upload)</div>', unsafe_allow_html=True)
        batch_file = st.file_uploader("Upload a CSV with the same feature columns", type=["csv"], key="batch")
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            batch_df.columns = [c.strip().replace("-", ".") for c in batch_df.columns]
            batch_df.replace("?", np.nan, inplace=True)

            feature_names = st.session_state["feature_names"]
            for feat in feature_names:
                if feat not in batch_df.columns:
                    batch_df[feat] = 0
            batch_input = batch_df[feature_names]

            preds = model.predict(batch_input)
            probas = model.predict_proba(batch_input)[:, 1]
            batch_df["Predicted_Income"] = [">50K" if p == 1 else "≤50K" for p in preds]
            batch_df["Confidence"] = (probas * 100).round(2)

            st.success(f"✅ Predicted {len(batch_df)} records")
            st.dataframe(batch_df.head(50), use_container_width=True)

            csv_out = batch_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Predictions CSV", csv_out,
                               file_name="predictions.csv", mime="text/csv",
                               use_container_width=True)
