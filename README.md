# 📊 Enhanced Stock Investment Portfolio Formation Using DBSCAN Clustering on S&P 500 Daily Returns

**Author:** Thavindu Liyanage  
**Student ID:** W1899297 / 20211175

---

## 🎯 Objective

This project explores and evaluates the effectiveness of three AI-based clustering techniques—**K-Means**, **DBSCAN**, and **Self-Organizing Maps (SOM)**—for forming optimized stock portfolios using **daily returns of S&P 500 stocks**. The goal is to determine the most effective method for:
- Grouping similar-performing stocks
- Managing outliers
- Enhancing portfolio performance with machine learning

---

## 📚 Application Area Overview

Machine learning is transforming how financial analysts and investors construct and optimize portfolios. Traditional methods like **K-Means** are commonly used, but this study emphasizes **DBSCAN** as a superior alternative due to its ability to:
- Identify noise/outliers
- Form arbitrarily-shaped clusters
- Eliminate the need to predefine the number of clusters

Additionally, **Artificial Neural Networks (ANNs)**—specifically **Self-Organizing Maps (SOMs)**—are evaluated for their non-linear pattern recognition capabilities in complex financial datasets.

---

## 📈 Data and Features

### 📅 Timeframe
- 2013–2023

### 📦 Data Sources
- **[Wikipedia S&P 500 List](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)**
- **[Yahoo Finance API (`yfinance`)](https://pypi.org/project/yfinance/)**

### 🔍 Features Used
- **Garman-Klass Volatility** – Robust estimator for stock price volatility  
- **RSI (Relative Strength Index)** – Measures price momentum  
- **Bollinger Bands** – Volatility-based envelope of moving averages  
- **ATR (Average True Range)** – Measures market volatility  
- **MACD (Moving Average Convergence Divergence)** – Trend-following momentum indicator

These indicators form the foundation for clustering and analysis.

---

## 🧠 AI Techniques Compared

| Clustering Model | Pros | Cons | Best Use Case |
|------------------|------|------|----------------|
| **K-Means** | Fast, scalable, easy to implement | Assumes spherical clusters, ignores outliers | Works best with well-defined, spherical clusters |
| **DBSCAN** | Handles noise and arbitrary shapes, no need to predefine cluster count | Sensitive to `eps` and `min_samples`, struggles with varying densities | Ideal for irregular cluster shapes and noisy data |
| **SOM (Self-Organizing Maps)** | Handles non-linear patterns, detects outliers, reduces dimensionality | Computationally expensive, requires large datasets, sensitive to hyperparameters | Suitable for complex, high-dimensional datasets |

---

## 🛠️ Implementation

### 📌 Tools & Libraries
- `Python`
- `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- `scikit-learn`
- `yfinance`
- `MiniSom` (for SOM implementation)
- `Draw.io` (for system diagrams)

### 🧮 Methodology
1. **Data Collection**: Gathered daily returns, technical indicators, and volatility metrics for all S&P 500 companies.
2. **Preprocessing**: Cleaned, normalized, and transformed data for clustering models.
3. **Modeling**:
    - Applied **K-Means** and evaluated with Elbow/Silhouette methods.
    - Applied **DBSCAN** using domain-optimized parameters.
    - Trained **Self-Organizing Map** to visualize and detect hidden stock group patterns.
4. **Evaluation**: Compared clustering results using visualizations, interpretability, noise handling, and application to portfolio diversification.

