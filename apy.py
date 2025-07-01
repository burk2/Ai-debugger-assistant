import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Model Debugger Assistant", layout="wide")

# Use dark background for plots
sns.set_style("darkgrid")
plt.style.use('dark_background')

st.title("🧠 AI Model Debugger Assistant")
st.markdown("""
Upload your **dataset** (CSV) and/or **training log**, and this assistant will analyze your data, diagnose problems, and recommend machine learning solutions.
""")

# -----------------------------
# 📥 Upload Dataset Section
# -----------------------------
st.header("📥 Upload Dataset")
data_file = st.file_uploader("Upload a dataset file (.csv)", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("🧹 Data Quality Check")
    st.write(f"🔢 Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Missing values
    st.write("🧼 Missing values per column:")
    st.write(df.isnull().sum())

    # Duplicates
    if df.duplicated().sum() > 0:
        st.warning(f"⚠️ Found {df.duplicated().sum()} duplicate rows.")

    # Select target
    target_col = st.selectbox("🎯 Select your target column", df.columns)

    # Detect task type
    if df[target_col].dtype == 'object' or df[target_col].nunique() <= 10:
        task_type = "classification"
    else:
        task_type = "regression"

    st.success(f"Detected Task Type: **{task_type.upper()}**")

    # Class distribution
    if task_type == "classification":
        st.subheader("⚖️ Class Distribution")
        class_counts = df[target_col].value_counts()
        st.bar_chart(class_counts)

        ratio = class_counts.max() / class_counts.min()
        if ratio > 1.5:
            st.warning("⚠️ Class imbalance detected. Consider SMOTE or class weights.")

    # Algorithm suggestions
    st.subheader("🧠 Suggested ML Algorithms")
    if task_type == "classification":
        st.markdown("""
        - Logistic Regression  
        - Random Forest Classifier  
        - XGBoost Classifier  
        - Support Vector Machine (SVM)
        """)
    else:
        st.markdown("""
        - Linear Regression  
        - Random Forest Regressor  
        - XGBoost Regressor  
        - Support Vector Regressor (SVR)
        """)

    # Visualizations
    st.subheader("📈 Target Distribution")
    fig1, ax1 = plt.subplots()
    if task_type == "classification":
        sns.countplot(x=target_col, data=df, palette="muted", ax=ax1)
    else:
        sns.histplot(df[target_col], kde=True, color='white', ax=ax1)
    st.pyplot(fig1)

    # Correlation heatmap (numeric only)
    st.subheader("🧪 Correlation Matrix")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) >= 2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

# -----------------------------
# 🧪 Upload Training Log Section
# -----------------------------
st.header("🧪 Upload Training Log")
log_file = st.file_uploader("Upload training log (.csv with columns: epoch, loss, val_loss, accuracy, val_accuracy)", type=["csv"], key="log")

if log_file:
    log_df = pd.read_csv(log_file)

    st.subheader("📄 Training Log Preview")
    st.dataframe(log_df.head())

    # Plot loss
    st.subheader("📉 Loss Curve")
    fig3, ax3 = plt.subplots()
    ax3.plot(log_df['epoch'], log_df['loss'], label='Training Loss')
    ax3.plot(log_df['epoch'], log_df['val_loss'], label='Validation Loss')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()
    st.pyplot(fig3)

    # Plot accuracy
    if 'accuracy' in log_df.columns and 'val_accuracy' in log_df.columns:
        st.subheader("📈 Accuracy Curve")
        fig4, ax4 = plt.subplots()
        ax4.plot(log_df['epoch'], log_df['accuracy'], '--', label='Training Accuracy', color='lime')
        ax4.plot(log_df['epoch'], log_df['val_accuracy'], '--', label='Validation Accuracy', color='magenta')
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy")
        ax4.legend()
        st.pyplot(fig4)

    # Diagnosis
    st.subheader("🧠 Model Behavior Diagnosis")

    try:
        acc_gap = log_df['accuracy'].iloc[-1] - log_df['val_accuracy'].iloc[-1]
        if acc_gap > 0.2:
            st.error("🚨 Overfitting Detected: Training accuracy much higher than validation.")
            st.markdown("**Fix Suggestions:** Add dropout, regularization, early stopping, or collect more data.")
        elif log_df['accuracy'].iloc[-1] < 0.6 and log_df['val_accuracy'].iloc[-1] < 0.6:
            st.warning("⚠️ Underfitting: Model is not learning enough.")
            st.markdown("**Fix Suggestions:** Use a deeper model, better features, or more training time.")
        elif log_df['loss'].diff().abs().mean() < 0.01:
            st.warning("⚠️ Loss is not improving — training may be stuck.")
            st.markdown("**Fix Suggestions:** Try a different optimizer or change learning rate.")
        else:
            st.success("✅ Training looks good. No major issues detected.")
    except Exception:
        st.info("⚠️ Could not compute diagnosis — make sure log columns are correct.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🛠️ Built by Nolin | AI/ML Debugger Assistant — v1.0")
