# 🏠 House Price Prediction using Machine Learning

A regression project predicting house prices using the Ames Housing Dataset with 90.6% accuracy.

---

## 🎯 Project Overview

**Objective:** Build a machine learning model to predict house sale prices  
**Dataset:** Ames Housing Dataset (1,460 samples, 80 features)  
**Best Model:** XGBoost Regressor  
**Accuracy:** R² = 0.9059 (90.59%)

---

## 📊 Results

### Model Performance Comparison

| Model | R² Score | RMSE ($) | Training Time (sec) |
|-------|----------|----------|---------------------|
| **XGBoost** ✅ | **0.9059** | **25,680** | 0.400 |
| Random Forest | 0.8978 | 26,761 | 1.154 |
| Linear Regression | 0.8630 | 30,979 | 0.022 |
| Decision Tree | 0.7971 | 37,701 | 0.024 |

**Interpretation:** The model predicts house prices within ±$25,680 on average, explaining 90.59% of price variance.

---

## 🔧 Features Used

**40 Total Features:**

**Numerical (10):**
- OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath
- YearBuilt, YearRemodAdd, MasVnrArea, Fireplaces, BsmtFinSF1

**Categorical (3, encoded to 30 columns):**
- Neighborhood (25 locations)
- ExterQual (Exterior quality)
- KitchenQual (Kitchen quality)

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas, numpy** - Data manipulation
- **scikit-learn** - ML models & preprocessing
- **XGBoost** - Gradient boosting
- **matplotlib, seaborn** - Visualization

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Run the Notebook
```bash
jupyter notebook notebook.ipynb
```

### Dataset
Download from: [Kaggle - House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Place `train.csv` and `test.csv` in the `data/` folder.

---

## 📈 Key Insights

1. **Top predictors:** Overall quality, living area, and neighborhood strongly influence price
2. **Combining numerical + categorical features** improved accuracy from 89% → 91%
3. **XGBoost outperformed simpler models** while maintaining fast training time
4. **Feature selection** (40 vs 280+ possible features) prevented overfitting

---

## 📝 Methodology

1. **Exploratory Data Analysis** - Correlation analysis, distribution plots
2. **Feature Engineering** - Selected top 10 numerical + 3 categorical features
3. **Preprocessing** - One-hot encoding for categorical variables
4. **Model Training** - Trained 4 different regression models
5. **Evaluation** - Compared using R² and RMSE metrics

---

## 🎓 Learning Outcomes

- Feature selection using correlation analysis
- Handling categorical data with one-hot encoding
- Comparing multiple ML algorithms
- Model evaluation with proper metrics
- End-to-end ML project workflow

---

## 📫 Contact

**Your Name** - https://www.linkedin.com/in/mohammad-faraz-rajput-837809194 | farazrajput112@gmail.com 

---

**Status:** ✅ Completed - 2 November 2025
```

---

### **Step 3: Create requirements.txt**

**Create a new file `requirements.txt` with:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
