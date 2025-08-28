---

🫀 Heart Disease Predictor

📌 Overview

This is a **machine learning project** that predicts the likelihood of heart disease using patient health data. The project is implemented entirely in **Google Colab** inside a single Jupyter Notebook.

The dataset is preprocessed, a model is trained, and predictions are appended as a new column (`prediction`) in the dataframe.

 ⚙️ Features

* Implemented in **Google Colab** (no local setup needed).
* Data preprocessing: scaling, encoding categorical variables, feature engineering.
* Trained with ML algorithms like **Logistic Regression** and **Random Forest**.
* Evaluated with metrics like Accuracy, Precision, Recall, F1-score, ROC-AUC.
* Prediction results shown directly in the dataframe.

📂 Project Structure

```
Heart-Disease-Predictor/
│-- heart_dataset.csv        # Dataset
│-- Disease_Predictor.ipynb  # Main notebook (Google Colab)
│-- README.md                # Documentation
```

🛠️ Technologies Used

* Google Colab ☁️
* Python 🐍
* Pandas, NumPy (Data preprocessing)
* Scikit-learn (Machine Learning)
* Matplotlib / Seaborn (Visualization)
* Joblib (Model saving/loading, optional)

📊 Model Performance (example)

* Accuracy: **87.6%**
* Precision: **85.2%**
* Recall: **89.1%**
* F1-Score: **87.1%**
* ROC-AUC: **0.91**

🚀 How to Run in Google Colab

1. Open the notebook:

   * Upload the dataset (`heart_dataset.csv`).
   * Run all cells in `Disease_Predictor.ipynb`.
2. The final dataframe will include a `prediction` column:

   ```
   prediction = 1 → Heart disease likely  
   prediction = 0 → No heart disease  
   ```

📌 Dataset

* Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

🔮 Future Work

* Try additional algorithms (XGBoost, Neural Networks).
* Hyperparameter tuning for higher accuracy.
* Create a simple **Streamlit GUI** for predictions.

👨‍💻 Author

* DEVAMANI T
* IT Student @ UCEV College

---

