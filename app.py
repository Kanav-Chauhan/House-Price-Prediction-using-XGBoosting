import streamlit as st
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# =========================
# Load Model
# =========================
with open("house_price_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load California Housing Data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Updated CV scores from training
cv_scores = [0.83732156, 0.84684611, 0.83359346, 0.85557869, 0.84062187]
cv_mean = np.mean(cv_scores)

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="🏠 California House Price Prediction", layout="centered")
st.title("🏠 California House Price Prediction App")

# =========================
# PREDICTION SECTION FIRST
# =========================
tab1, tab2 = st.tabs(["🎯 Single Random Prediction", "📈 50 Random Predictions"])

# ---- Tab 1: Single prediction ----
with tab1:
    if st.button("🎯 Generate Random Prediction"):
        random_idx = random.randint(0, len(X) - 1)
        random_features = X.iloc[random_idx].to_frame().T
        actual_price = y.iloc[random_idx] * 100000
        predicted_price = pipeline.predict(random_features)[0] * 100000

        st.write("**Random House Features:**")
        st.dataframe(random_features)

        st.write(f"🏷 **Actual Price:** ${actual_price:,.2f}")
        st.write(f"🤖 **Predicted Price:** ${predicted_price:,.2f}")

        mae = mean_absolute_error([actual_price], [predicted_price])
        st.info(f"📏 Absolute Error: ${mae:,.2f}")

        if abs(actual_price - predicted_price) / actual_price < 0.1:
            st.success("✅ Pretty close prediction!")
        else:
            st.warning("⚠️ Prediction is off by more than 10%.")

# ---- Tab 2: Multiple predictions ----
with tab2:
    if st.button("📈 Run 50 Random Predictions"):
        sample_indices = random.sample(range(len(X)), 50)
        X_sample = X.iloc[sample_indices]
        y_sample_actual = y.iloc[sample_indices] * 100000
        y_sample_pred = pipeline.predict(X_sample) * 100000

        fig2, ax2 = plt.subplots()
        ax2.scatter(y_sample_actual, y_sample_pred, alpha=0.7)
        ax2.plot([y_sample_actual.min(), y_sample_actual.max()],
                 [y_sample_actual.min(), y_sample_actual.max()],
                 color="red", linestyle="--")
        ax2.set_xlabel("Actual Price ($)")
        ax2.set_ylabel("Predicted Price ($)")
        ax2.set_title("Actual vs Predicted Prices (50 Random Samples)")
        st.pyplot(fig2)

        mae_multi = mean_absolute_error(y_sample_actual, y_sample_pred)
        r2_multi = r2_score(y_sample_actual, y_sample_pred)
        st.write(f"📏 Mean Absolute Error (50 samples): ${mae_multi:,.2f}")
        st.write(f"📊 R² Score (50 samples): {r2_multi:.4f}")

# =========================
# XGBOOST SECTION
# =========================
st.subheader("🔍 Why XGBoost Regressor?")
st.write("""
- XGBoost is one of the most powerful and efficient gradient boosting algorithms.
- It handles **non-linear relationships** and **feature interactions** exceptionally well.
- It can manage missing data, prevent overfitting using regularization, and scale to large datasets.
""")

# =========================
# K-FOLD SECTION
# =========================
st.subheader("📚 Why K-Fold Cross Validation?")
st.write("""
- Train-test split evaluates the model on only one subset of the data, which can cause **biased performance estimates**.
- K-Fold CV trains and tests the model **multiple times** on different folds of the data, giving a **more reliable and robust accuracy**.
- This ensures our model generalizes better to unseen data.
""")

# =========================
# MODEL PERFORMANCE & GRAPH
# =========================
st.subheader("📊 Model Performance (from training)")
st.write(f"Average R² Score from 5-Fold CV: **{cv_mean:.4f}**")
st.write(f"R² Scores from each fold: {np.round(cv_scores, 4)}")

st.subheader("📌 Feature Importance")
model_only = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]  # auto-detect last step
importances = model_only.feature_importances_

fig, ax = plt.subplots()
ax.barh(X.columns, importances)
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Feature Importance from XGBoost")
st.pyplot(fig)

# =========================
# REAL WORLD APPLICATIONS
# =========================
st.subheader("🌍 Real-World Applications")
st.write("""
- **Real Estate Agencies**: Estimate property prices for sellers and buyers.
- **Investment Analysis**: Help investors decide which regions have the best ROI potential.
- **Urban Planning**: Understand how location and features influence housing costs.
- **Banking & Insurance**: Use in loan approvals and property risk assessment.
""")

st.markdown("""## Made with ❤️ By Kanav Chauhan""")
