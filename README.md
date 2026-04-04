📊 AI Data Insights Tool

AI-powered Python tool to analyze your CSV datasets, detect anomalies, predict missing values, and provide machine learning insights. Also comes with a Streamlit app for interactive usage.

🚀 Features
Basic Dataset Info – Rows, columns, missing values.

Correlation Insights – Detects strong correlations between features.

Anomaly Detection – Flags unusual data points using Isolation Forest.
Train ML Model – Random Forest (classification or regression) to predict target variables.
Fill Missing Values – Automatically fills missing fields in the dataset using ML.
Predict Single Missing Field – If you leave exactly one field blank, it predicts it based on other values.
Feature Importance – Shows top factors influencing the target variable.
Final Summary Report – AI-generated insights including dataset reliability, correlations, anomalies, and model performance.

🗂 Repository Structure
├── ml_data_analysis.py       # Main engine containing all functions
├── app.py                    # Streamlit interactive application
├── requirements.txt          # Python dependencies
└── README.md                 # This guide

⚙️ Setup Instructions
1. Clone the Repository
 git clone https://github.com/WALKER00058/ML-data-analysis/
 cd ML-data-analysis
2. Install Dependencies
 Make sure you have Python 3.10+ installed. Then run:
  pip install -r requirements.txt
3. Prepare Dataset
 Use any CSV file.
 The engine can automatically detect numeric and categorical features.
 Missing values can be handled, but the CSV should be correctly formatted.
 No need to stick with Titanic data(used for initial testing by developer)—the engine works with any dataset.

🖥 Running the Engine
You can run the Python engine directly:
 python ml_data_analysis.py
The engine will:
 1.Load your CSV file (default: sample.csv or replace path inside the script).
 2.Display basic info, correlation insights, anomalies.
 3.Train a Random Forest model and show metrics (accuracy, precision, recall, F1).
 4.Fill missing values and provide predictions for single missing fields.
 5.Generate a final AI-powered summary.

🌐 Running the Streamlit App
In VS Code terminal, run : streamlit run app.py
App Features:
 Upload your CSV file directly.
 Preview your dataset.
 Run analysis with one click.
 Fill missing data in any column.
 Predict a single missing field by leaving it blank.
 Download the updated CSV with filled values.
 View ML model performance and feature importance.
 See the AI-generated final summary.

⚠️ Known Issues / Notes
1. Leave exactly one field blank for predict_missing_field to work.
2. Extremely small datasets (<20 rows) may give less reliable insights.
3. ML model performance depends on dataset quality and features. Accuracy may vary.
4. Ensure numeric fields are properly formatted; text in numeric columns may break model training.
5. Built-in libraries like os, warnings, logging are standard Python—no need to install.
6. Report bugs or unexpected behavior via GitHub issues.

📌 Best Practices
Always backup your CSV before filling missing values.
Use a consistent dataset structure for repeated predictions.
For better predictions, try larger datasets with clean numeric and categorical features.
Explore feature_importance to understand which fields influence the target most.

🛠 Dependencies (requirements.txt)
pandas
numpy
scikit-learn
joblib
treamlit
logging

📝 Contributing
Report bugs via GitHub issues.
Pull requests for improvements are welcome.
Keep dataset-agnostic behavior in mind when changing functions.

Credits 
Developed as a personal AI-powered data analysis engine. Inspired by classic machine learning workflows and interactive dashboard design.

Feel free to modify, experiment, and improve this project! 🛠️
If you make any changes, have ideas, or spot bugs, share them in the GitHub comments or issues—it really helps me grow as a developer and make the tool even better.
