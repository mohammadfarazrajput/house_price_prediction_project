````md
# 🏠 House Price Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)

> **Predict house prices with 90.6% accuracy using XGBoost. Systematic feature engineering reduces 81 features to 13, achieving top 25% Kaggle performance.**

---

## 🎯 Project Overview

Built an end-to-end ML pipeline to predict house sale prices on the **Ames Housing Dataset**. After preprocessing 1,460 samples and engineering 13 features from 81 original ones, **XGBoost achieved 90.6% R² accuracy** with $25,680 RMSE (14.2% error rate).

**Key Achievements:**
- ✅ 90.6% test accuracy (matches professional appraisal accuracy of ±10–15%)
- ✅ Reduced features from 81 → 13 through correlation analysis
- ✅ Top 25% Kaggle leaderboard equivalent
- ✅ Production-ready model in 0.27 seconds training time

---

## 📊 Results

### Model Performance

| Model | Test R² | RMSE ($) | Train Time (s) |
|-------|---------|----------|----------------|
| **XGBoost** ✅ | **0.9059** | **25,680** | 0.27 |
| Random Forest | 0.8978 | 26,761 | 1.21 |
| Linear Regression | 0.8630 | 30,979 | 0.02 |
| Decision Tree | 0.7971 | 37,701 | 0.03 |

**Overfitting Check:**
- Train R²: 0.9870 | Test R²: 0.9195 | Gap: 6.75% ✅ (Healthy)

---

## 🔧 Features Used (13 Total)

**Numerical (10):** `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `FullBath`, `YearBuilt`, `YearRemodAdd`, `MasVnrArea`, `Fireplaces`, `BsmtFinSF1`

**Categorical (3):** `Neighborhood` (25 locations), `ExterQual` (exterior quality), `KitchenQual` (kitchen quality)

→ **40 columns after one-hot encoding**

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
````

### Download Dataset

```bash
# Using Kaggle CLI
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip
```

Or download manually from: [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

### Run the Project

```bash
# Start Jupyter Notebook
jupyter notebook house_price_prediction.ipynb

# Or run Python script
python train_model.py
```

### Use Saved Model for Predictions

```python
import pickle
import pandas as pd

# Load model
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new house data (40 features after encoding)
new_house = {...}  # Your feature dictionary
prediction = model.predict(pd.DataFrame([new_house]))
print(f"Predicted Price: ${prediction[0]:,.2f}")
```

-----

## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   ├── train.csv                 # Training dataset
│   └── test.csv                  # Test dataset
│
├── models/
│   └── xgboost_model.pkl         # Saved best model
│
├── plots/
│   ├── feature_importance.png    # Top features visualization
│   ├── actual_vs_predicted.png   # Prediction accuracy
│   └── residual_analysis.png     # Error distribution
│
├── house_price_prediction.ipynb  # Main notebook
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

-----

## 🔍 Methodology

1.  **EDA:** Analyzed 1,460 houses with 81 features
2.  **Feature Engineering:**
      * Removed 18 low-correlation features (\<0.2)
      * Dropped 19 features with \>5% missing values
      * Eliminated 4 multicollinear features (\>0.8 correlation)
      * Selected 13 high-impact features
3.  **Preprocessing:** One-hot encoding + StandardScaler
4.  **Training:** Compared 4 algorithms (Linear, Tree, RF, XGBoost)
5.  **Validation:** 80-20 train-test split, overfitting analysis

-----

## 💡 Key Insights

  * **Quality \> Size:** `OverallQual` is the \#1 predictor (stronger than `GrLivArea`)
  * **Location Matters:** `Neighborhood` creates 20–30% price variance
  * **Categorical Features:** Adding 3 categorical features improved R² from 88.5% → 90.6% (+2.1%)
  * **Model Choice:** XGBoost outperforms simpler models while maintaining fast training

**Top 3 Features:**

1.  `OverallQual` (0.79 correlation)
2.  `GrLivArea` (0.71 correlation)
3.  `GarageCars` (0.64 correlation)

-----

## 🛠️ Technologies

  * **Python 3.8+** - Programming language
  * **Pandas/NumPy** - Data manipulation
  * **Scikit-learn** - ML models & preprocessing
  * **XGBoost** - Gradient boosting
  * **Matplotlib/Seaborn** - Visualization
  * **Jupyter** - Interactive development

-----

## 🚧 Future Improvements

  * Hyperparameter tuning with GridSearchCV (expected +1–2% R²)
  * Advanced feature engineering (polynomial features, interactions)
  * Ensemble stacking (XGBoost + Random Forest)
  * Deploy as Flask/FastAPI web service
  * Add external data (school ratings, crime stats, walkability)

-----

## 📈 Visualizations

**Feature Importance:**

*Shows OverallQual, GrLivArea, and GarageCars as top predictors*

**Actual vs Predicted:**

*Tight clustering around diagonal shows strong prediction accuracy*

**Residual Analysis:**

*Normal distribution centered at 0 confirms model is unbiased*

*(See `/plots/` folder for all visualizations)*

-----

## 📝 License

This project is licensed under the MIT License - see `LICENSE` file for details.

-----

## 📫 Contact

**Mohammad Faraz Rajput**

📧 **Email:** `farazrajput112@gmail.com`  
💼 **LinkedIn:** `linkedin.com/in/mohammad-faraz-rajput-837809194`  
🐙 **GitHub:** `@yourusername`

## ⭐ *If you found this project helpful, please star the repository\!*

**Status:** ✅ Complete | November 2025

```
```
