import streamlit as st
import pandas as pd

from ml_data_analysis import (
    fill_all_missing_values,
    load_data,
    basic_info,
    correlation_insights,
    detect_anomalies,
    predict_missing_field,
    train_model,
    generate_summary
)

st.set_page_config(page_title="AI Data Insights Tool", layout="wide")
st.title("📊 AI Data Insights Tool")
st.write("Upload your dataset and get instant AI-powered analysis.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
corr = []
anomalies = []
model_info = None

if uploaded_file is not None:

    if "original_df" not in st.session_state:
        st.session_state["original_df"] = pd.read_csv(uploaded_file)

    if "current_df" not in st.session_state:
        st.session_state["current_df"] = st.session_state["original_df"].copy()

    df = st.session_state["current_df"] 
    if "original_df" not in st.session_state:
     st.session_state["original_df"] = df.copy()
    if "current_df" not in st.session_state:
     st.session_state["current_df"] = df.copy()
    df = st.session_state["current_df"]
    if st.button("Reset Dataset"):
     st.session_state["current_df"] = st.session_state["original_df"].copy()
     st.success("Dataset reset successfully!")
    st.subheader("📄 Dataset Preview")

    st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    orig_df = st.session_state["original_df"]
    show_full = st.checkbox("Show full dataset", key="original_dataset")

    if show_full:
     st.dataframe(orig_df)
    else:
     st.dataframe(orig_df.head(10))

    st.markdown("### 📊 Analysis")
    

    if st.button("Run Analysis"):
     st.session_state["analysis_done"] = True
     st.session_state["corr"] = correlation_insights(df)
     st.session_state["anomalies"] = detect_anomalies(df)
     st.session_state["basic_info"] = basic_info(df)
    if st.session_state.get("analysis_done", False):

     st.subheader("📌 Basic Info")
     st.write(st.session_state["basic_info"])

     st.subheader("🔗 Correlation Insights")
     for item in st.session_state["corr"]:
         st.write("•", item)

     st.subheader("🚨 Anomalies")
     anomalies = st.session_state["anomalies"]

     if anomalies:
         st.write(f"{len(anomalies)} anomalies found")
         st.write(anomalies[:3])
     else:
         st.write("No anomalies found")

    st.markdown("### 🧩 Data Cleaning")

    if st.button("Fill ALL Missing Data"):
     with st.spinner("Filling missing values using AI..."):
         st.session_state["current_df"] = fill_all_missing_values(st.session_state["current_df"].copy())
         st.session_state["filled_clicked"] = True
     st.success("All missing values filled!")
     st.rerun()   
    if st.session_state.get("filled_clicked", False):
     
     filled_df = st.session_state["current_df"]
     

     show_full_filled = st.checkbox("Show full dataset after filling", key="filled_dataset")

     if show_full_filled:
         st.dataframe(filled_df)
     else:
         st.dataframe(filled_df.head(10))    

     csv = df.to_csv(index=False)
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

         input_data[col] = pred
         st.success(f"Predicted {col}: {pred}")

         new_row = pd.DataFrame([input_data])

         # align columns
         new_row = new_row.reindex(columns=df.columns, fill_value=None)

         updated_df = pd.concat([st.session_state["current_df"], new_row], ignore_index=True)

         st.session_state["current_df"] = updated_df
         st.session_state["prediction_clicked"] = True   
     else:
         st.error(result)
    if st.session_state.get("prediction_clicked", False):

         pred_df = st.session_state["current_df"]

         st.subheader("📊 Updated Dataset (after prediction)")

         st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

         show_full_pred = st.checkbox(
            "Show full dataset after prediction",
            key="prediction_dataset"
         )

         if show_full_pred:
             st.dataframe(df, use_container_width=True)
         else:
             st.dataframe(df.head(10), use_container_width=True)

         csv = df.to_csv(index=False)
         st.download_button(
            "Download Final CSV (includes filled + predicted data)",
            csv,
            "final_data.csv"
         )

    st.subheader("🤖 Model Performance")
    with st.spinner("Analyzing data..."):
        model_info = train_model(df)

    if model_info is not None:
        st.write(f"Accuracy: {model_info['accuracy']:.2f}")
        if model_info["type"] == "classification":
            st.write(f"Precision: {model_info['precision']:.2f}")
            st.write(f"Recall: {model_info['recall']:.2f}")
            st.write(f"F1 Score: {model_info['f1_score']:.2f}")

        elif model_info["type"] == "regression":
            st.write(f"MSE: {model_info['mse']:.2f}")
            st.write(f"RMSE: {model_info['rmse']:.2f}")
            st.write(f"R² Score: {model_info['r2']:.2f}")
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
    else:
        st.warning("Model training not available for this dataset. Ensure you have at least two numeric columns and a 'Survived' target column.")

        st.subheader("🧠 Final Insights")
        summary = generate_summary(df, corr, anomalies, None)
        for item in summary:
            st.write("•", item)