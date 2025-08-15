# 🏠 House Price Prediction using XGBoost - [Click for Live Demo](https://kanav-house-price-prediction.streamlit.app/)

This project is a **machine learning web app** built with **Streamlit** that predicts **California house prices** using the **California Housing Dataset** and an **XGBoost Regressor** model.

---

## 📌 Project Overview

Using real-world housing data, the app allows you to:
- Input house features manually and get a **predicted price**.
- Generate **random test samples** to see how accurate the model is.
- Run **bulk predictions (50 random houses)** to evaluate performance.
- View interactive charts for **data distribution and model evaluation**.

---

## ⚙️ Tech Stack

- **Python** – Core programming language
- **Pandas, NumPy** – Data processing
- **Scikit-learn** – Preprocessing & pipeline
- **XGBoost** – Regression model
- **Streamlit** – Web app framework
- **Matplotlib / Plotly** – Data visualization

---

## 📊 Features

✅ Predict house prices from user input  
✅ Generate **Random Predictions** with error comparison  
✅ Evaluate performance with **MAE** (Mean Absolute Error)  
✅ Display **distribution charts** for price trends  
✅ Fully interactive UI with **custom-styled buttons**

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kanav-Chauhan/House-Price-Prediction-using-XGBoosting.git
   cd house-price-prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```


---

## 📈 Model Details

- **Algorithm:** XGBoost Regressor
- **Dataset:** California Housing Dataset (from `sklearn.datasets`)
- **Evaluation Metric:** Mean Absolute Error (MAE)

The model is wrapped in a **Scikit-learn pipeline** with preprocessing steps for scaling and feature handling.

---

## 💡 Future Improvements

- Add more **feature engineering**
- Integrate **real-time housing market data**
- Provide **confidence intervals** for predictions
- Deploy online (Streamlit Cloud / Heroku / AWS)

---
## Made with ❤️ by Kanav Chauhan

