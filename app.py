import streamlit as st
import pandas as pd

from ml_data_analysis import (
    fill_missing_values,
    load_data,
    basic_info,
    correlation_insights,
    detect_anomalies,
    predict_missing_field,
    train_model,
    generate_summary
)
st.markdown("---")
st.set_page_config(page_title="AI Data Insights Tool", layout="wide")
st.title("📊 AI Data Insights Tool")
st.write("Upload your dataset and get instant AI-powered analysis.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### 📊 Analysis")
    column_to_fill = st.selectbox("Select column to fill", df.columns)

    if st.button("Run Analysis"):
        st.subheader("📌 Basic Info")
        st.write(basic_info(df))

        st.subheader("🔗 Correlation Insights")
        with st.spinner("Analyzing data..."):
            corr = correlation_insights(df)
        for item in corr:
            st.write("•", item)

        st.subheader("🚨 Anomalies")
        with st.spinner("Analyzing data..."):
            anomalies = detect_anomalies(df)
        if anomalies:
            st.write(f"{len(anomalies)} anomalies found")
            st.write(anomalies[:3])
        else:
            st.write("No anomalies found")

    st.markdown("### 🧩 Data Cleaning")
    if st.button("Fill Missing Data"):
        with st.spinner("Analyzing data..."):
            filled_df = fill_missing_values(df, column_to_fill)
        st.success("Missing values filled!")
        st.dataframe(filled_df.head())
        csv = filled_df.to_csv(index=False)
        st.download_button("Download Updated CSV", csv, "filled_data.csv")

    st.markdown("### 🔮 Prediction")
    st.subheader("Predict missing field (Fill n-1 fields)")
    input_data = {}
    for col in df.columns:
        if df[col].dtype == "object":
            value = st.selectbox(f"{col} (leave blank if unknown)", [""] + list(df[col].dropna().unique()))
            input_data[col] = None if value == "" else value
        else:
            value = st.text_input(f"{col} (leave blank if unknown)")
            try:
                input_data[col] = None if value == "" else float(value)
            except:
                st.error(f"Invalid input for {col}")
                input_data[col] = None

    if st.button("Predict Missing Field"):
        with st.spinner("Analyzing data..."):
            result = predict_missing_field(df, input_data)
        if isinstance(result, tuple):
            col, pred = result
            st.success(f"Predicted {col}: {pred}")
        else:
            st.error(result)

    st.subheader("🤖 Model Performance")
    with st.spinner("Analyzing data..."):
        model_info = train_model(df)
    st.write(f"Accuracy: {model_info['accuracy']:.2f}")
    if model_info["precision"] is not None:
        st.write(f"Precision: {model_info['precision']:.2f}")
        st.write(f"Recall: {model_info['recall']:.2f}")
        st.write(f"F1 Score: {model_info['f1_score']:.2f}")

    st.subheader("📊 Feature Importance")
    sorted_features = sorted(
        model_info["feature_importance"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for k, v in sorted_features[:10]:
        st.write(f"{k}: {v:.2f}")

    st.subheader("🧠 Final Insights")
    summary = generate_summary(df, corr, anomalies, model_info)
    for item in summary:
        st.write("•", item)