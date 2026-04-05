import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import logging
import joblib
import os
from sklearn.metrics import precision_score, recall_score, f1_score
logging.basicConfig(level=logging.WARNING)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. LOAD DATA
# -----------------------------
def load_data(path):
    try:
        df = pd.read_csv(path)
        if df.empty:
            logging.warning(f"Warning: {path} is empty")
        return df
    except FileNotFoundError:
        logging.error(f"Error: File {path} not found")
        raise
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        raise


# ----------------------------- 
# 2. BASIC ANALYSIS
# -----------------------------
def basic_info(df):
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict()
    }
    return info


# -----------------------------
# 3. CORRELATION INSIGHTS
# -----------------------------
def correlation_insights(df):
    df_processed = preprocess_data(df)
    corr = df_processed.corr()
    insights = []
    seen = set()

    max_insights = 5
    extra_flag = False

    for col in corr.columns:
        for row in corr.index:
            pair = tuple(sorted([row, col]))

            if col != row and pair not in seen and abs(corr.loc[row, col]) > 0.7:
                if len(insights) < max_insights:
                    insights.append(f"{row} and {col} are strongly related ({corr.loc[row, col]:.2f})")
                    seen.add(pair)
                else:
                    extra_flag = True
                    break   # break inner loop

        if extra_flag:
            break   # break outer loop

    if extra_flag:
        insights.append("...and more relationships exist")

    return insights


# -----------------------------
# 4. ANOMALY DETECTION
# -----------------------------
def detect_anomalies(df):
    df_copy = df.copy()

    # preprocess entire dataset (no target splitting needed)
    numeric_df = preprocess_data(df_copy)

    if numeric_df.shape[1] == 0:
        logging.warning("No numeric columns found for anomaly detection")
        return []

    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(numeric_df)

    anomalies = df_copy[preds == -1]
    return anomalies.to_dict(orient="records")


# -----------------------------
# 5. TRAIN MODEL + FEATURE IMPORTANCE
# -----------------------------
def preprocess_data(df):
    drop_cols = ["Name", "Ticket", "Cabin"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # numeric
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # categorical
    cat_cols = df.select_dtypes(exclude=np.number).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    df = pd.get_dummies(df, drop_first=True)

    return df
def train_model(df, target_col=None):
    df_copy = df.copy()

    #  select target
    if target_col is None:
        target = df_copy.columns[-1]
    else:
        target = target_col

    X_raw = df_copy.drop(columns=[target])
    y = df_copy[target]

    # remove missing target rows
    valid_idx = y.notna()
    X_raw = X_raw[valid_idx]
    y = y[valid_idx]

    # preprocess X
    X = preprocess_data(X_raw)

    # 🧠 better problem detection
    if y.dtype == "object":
        problem_type = "classification"
    elif y.nunique() < 15:
        problem_type = "classification"
    else:
        problem_type = "regression"

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # model
    if problem_type == "classification":
        model = RandomForestClassifier(random_state=42, max_depth=5, min_samples_leaf=3)
    else:
        model = RandomForestRegressor(random_state=42, max_depth=5, min_samples_leaf=3)

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # metrics
    if problem_type == "classification":
        accuracy = model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        mse = rmse = r2 = None

    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        accuracy = r2   # 👈 better interpretation
        precision = recall = f1 = None

    importance = dict(zip(X.columns, model.feature_importances_))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "feature_importance": importance,
        "target": target,
        "type": problem_type
    }

# -----------------------------
# 6. FILL MISSING VALUES FUNCTION
# -----------------------------
def fill_all_missing_values(df):
    df_filled = df.copy()

    for col in df.columns:
        if df_filled[col].isna().sum() == 0:
            continue

        known = df_filled[df_filled[col].notna()]
        missing = df_filled[df_filled[col].isna()]

        if missing.empty or known.shape[0] < max(2, int(0.1 * len(df))):
            continue

        X_known = known.drop(columns=[col])
        y_known = known[col]

        X_missing = missing.drop(columns=[col])

        # Encode ONLY features
        X_known_encoded = pd.get_dummies(X_known)
        X_missing_encoded = pd.get_dummies(X_missing)
        valid_idx = X_known_encoded.notna().all(axis=1)
        X_known_encoded = X_known_encoded[valid_idx]
        y_known = y_known[valid_idx]
        X_missing_encoded = X_missing_encoded.fillna(0)

        # Align columns
        X_missing_encoded = X_missing_encoded.reindex(columns=X_known_encoded.columns,fill_value=0)
        X_known_encoded = X_known_encoded.fillna(X_known_encoded.median())

        # Model selection
        if y_known.dtype == "object" or y_known.nunique() <= 10:
            model = RandomForestClassifier(random_state=42, max_depth=5, min_samples_leaf=3)
        else:
            model = RandomForestRegressor(random_state=42, max_depth=5, min_samples_leaf=3)

        model.fit(X_known_encoded, y_known)

        preds = model.predict(X_missing_encoded)

        df_filled.loc[df_filled[col].isna(), col] = preds

    return df_filled
# -----------------------------
# 7. PREDICTION FUNCTION
# -----------------------------
def predict_missing_field(df, input_dict):
    df_copy = df.copy()

    # Find missing field
    missing_cols = [k for k, v in input_dict.items() if v is None]

    if len(missing_cols) != 1:
        return "Please leave exactly ONE field empty."

    target_col = missing_cols[0]

    # ✅ Check in ORIGINAL df (NOT processed)
    if target_col not in df_copy.columns:
        return "Invalid column."

    # Split features/target BEFORE preprocessing
    X_raw = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]

    # Preprocess AFTER split
    X = preprocess_data(X_raw)
    # remove rows where target is missing
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    # Choose model
    if y.nunique() <= 10:
        model = RandomForestClassifier(random_state=42, max_depth=5, min_samples_leaf=3)
    else:
        model = RandomForestRegressor(random_state=42, max_depth=5, min_samples_leaf=3)

    model.fit(X, y)
    

    # Prepare input
    input_df = pd.DataFrame([input_dict])

    input_X_raw = input_df.drop(columns=[target_col])
    input_processed = preprocess_data(input_X_raw)
    input_processed = input_processed.fillna(0)

    # Align columns
    input_processed = input_processed.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_processed)

    return target_col, prediction[0]
# -----------------------------
# 8. GENERATE FINAL INSIGHTS
# -----------------------------
def generate_summary(df, corr, anomalies, model_info):
    summary = []

    # dataset size warning
    if len(df) < 20:
        summary.append("⚠️ Dataset is very small. Insights may not be reliable.")

    # correlation reliability warning
    if len(corr) > 0 and len(df) < 10:
        summary.append("High correlations detected, but may be exaggerated due to small dataset size.")

    if corr and len(corr) > 0:
        summary.append(f"{len(corr)} strong relationships were found in the given data.")
    else:
        summary.append("No strong correlations detected.")

    if anomalies and len(anomalies) > 0:
        summary.append(f"{len(anomalies)} unusual data points detected.")
    else:
        summary.append("No anomalies detected.")

    if model_info:
        summary.append(f"A prediction model was trained with {model_info['accuracy']:.2f} accuracy.")
        important = sorted(model_info["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        top_features = [f"{feat}" for feat, _ in important[:3]]
        summary.append(f"Top factors affecting '{model_info['target']}': {', '.join(top_features)}.")

    else:
        summary.append("Model training failed or insufficient data.")
    missing_total = sum(df.isnull().sum())
    if missing_total > 0:
        summary.append(f"Dataset contains {missing_total} missing values. Consider cleaning the data.")
    if model_info and model_info["accuracy"] > 0.7:
        summary.append("Overall: The dataset shows strong predictive potential.")
    elif model_info:
        summary.append("Overall: Predictions are weak — more data or better features may be needed.")
    if model_info and model_info["type"] == "classification":
        summary.append(
            f"Model performance → Accuracy: {model_info['accuracy']:.2f}, "
            f"Precision: {model_info['precision']:.2f}, "
            f"Recall: {model_info['recall']:.2f}, "
            f"F1 Score: {model_info['f1_score']:.2f}."
        )
    elif model_info and model_info["type"] == "regression":
        summary.append(
            f"Model performance → MSE: {model_info['mse']:.2f}, "
            f"RMSE: {model_info['rmse']:.2f}, "
            f"R² Score: {model_info['r2']:.2f}."
        )
    summary.append("This analysis was automatically generated using an AI-powered data insight engine.")
    return summary
# -----------------------------
# MAIN RUN
# -----------------------------
if __name__ == "__main__":
    try:
        # Parameterize file path
        data_path = "sample.csv"
        df = load_data(data_path)
        
        print("\n=== BASIC INFO ===")
        print(basic_info(df))

        print("\n=== CORRELATION INSIGHTS ===")
        corr = correlation_insights(df)
        for item in corr:
         print(f"• {item}")

        print("\n=== ANOMALIES ===")
        anomalies = detect_anomalies(df)
        if anomalies:
         print(f"{len(anomalies)} anomalies found. Example:")
         print(anomalies[0])
        else:
         print("No anomalies found")

        print("\n=== MODEL INFO ===")
        model_info = train_model(df)  # Optional: specify target_col parameter
        print(f"Accuracy: {model_info['accuracy']:.2f}")
        if model_info['precision'] is not None:
         print(f"Precision: {model_info['precision']:.2f}")
         print(f"Recall: {model_info['recall']:.2f}")
         print(f"F1 Score: {model_info['f1_score']:.2f}")
        

        print("\n=== RAW MODEL INFO (debug) ===")
        print(model_info)
        print("\n=== FEATURE IMPORTANCE ===")
        sorted_features = sorted(model_info["feature_importance"].items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_features[:10]:
         print(f"{k}: {v:.2f}")
        print("\n=== FINAL SUMMARY ===")
        summary = generate_summary(df, corr, anomalies, model_info)
        for item in summary:
            print(f"  • {item}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")
        