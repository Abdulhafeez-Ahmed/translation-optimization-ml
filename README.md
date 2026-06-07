# Dual-Engine Translation Workflow Optimization Pipeline

A production-ready machine learning workspace implementing **Multiple Linear Regression** and **Binary Logistic Regression** entirely from scratch using **NumPy** and **Pandas**. 

This repository simulates an automated project triaging ecosystem for language service providers (LSPs). By analyzing a document's deep structural and semantic footprint, the pipeline automatically answers two critical operational questions:
1. **The Regression Engine:** *How many labor minutes will this document take to translate?*
2. **The Classification Engine:** *Is the technical/formatting difficulty level Easy (0) or Hard (1)?*

---

## 📊 Core Workspace Architecture

The framework is explicitly structured to mimic industry-standard modular software engineering patterns. Rather than relying on black-box external machine learning libraries (such as scikit-learn), all optimization equations, gradient computations, and normalization pipelines were built by hand.

---

## 🛠️ Mathematical Implementation Details (`utils.py`)

The mathematical engine contains highly optimized, vectorized linear algebra operations using NumPy array broadcasting. This approach bypasses slow Python `for` loops, enabling computations to run at hardware speed.

### 1. Data Normalization Pipeline
To prevent massive feature scales (like a 5,000 word count) from numerically drowning out small fractional ratios (such as a 12% technical term density), the module implements **Z-Score Standardization**:

$$\mu = \frac{1}{m} \sum_{i=1}^{m} X_i, \quad \sigma = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (X_i - \mu)^2}, \quad X_{\text{norm}} = \frac{X - \mu}{\sigma}$$

### 2. Regression Engine (Continuous Prediction)
* **Objective Function:** Tailored around **Mean Squared Error (MSE)** loss:
  $$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(X^{(i)}) - y^{(i)} \right)^2$$
* **Gradient Descent:** Simulates partial derivatives tracking parameter slopes:
  $$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (Xw + b - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (Xw + b - y)$$

### 3. Classification Engine (Discrete Triage)
* **Activation Function:** Implements the classic nonlinear **Sigmoid squashing function** to map raw scores into bounded probabilities $P \in (0, 1)$:
  $$g(z) = \frac{1}{1 + e^{-z}}$$
* **Objective Function:** Structured around **Binary Cross-Entropy Loss** to apply strong logarithmic penalties to confident, incorrect classifications:
  $$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{w,b}(X^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(X^{(i)})) \right]$$

---

## 📈 Dataset Structural Footprint (`translation_data.csv`)

The dataset contains a carefully engineered multi-dimensional matrix of structural text characteristics across 15 production configurations:

| Feature Dimension | Data Type | Description |
| :--- | :--- | :--- |
| **`words`** | Continuous | Total absolute text volume (word count). |
| **`avg_sentence_length`** | Continuous | Average words per sentence (syntactic density tracker). |
| **`sentence_complexity_pct`** | Percentage | Ratio of subordinate/complex clauses to simple clauses. |
| **`technical_terms_pct`** | Percentage | Density of domain-specific glossary/specialized terminology. |
| **`translation_time_min`** | Continuous | **Regression Target:** Exact labor time required (minutes). |
| **`difficulty`** | Binary (0 / 1) | **Classification Target:** Layout & formatting complexity (Easy vs Hard). |

---

## 🚀 Key Performance Dashboards & Results

### 1. Regression Model (Labor Estimation)
* **Convergence Status:** Successfully stabilized with a final bias scalar ($b$) of `497.00`.
* **Feature Weights Insights:**
  * `words`: **+195.70** (The strongest driver of absolute labor time).
  * `technical_terms_pct`: **+131.04** (Highly dense terminology significantly inflates time).
  * `sentence_complexity_pct`: **-80.99** (Indicates an inverse semantic alignment when interacting with vocabulary baselines).

### 2. Classification Model (Difficulty Triaging)
* **Training Accuracy:** **100.0% Perfect Linear Separation** achieved on the 15-document baseline.
* **Operational Triage Log:**

| Actual Class | Predicted Probability | Predicted Class | Operational Status |
| :--- | :--- | :--- | :--- |
| 0 (Easy) | 0.0% | 0 | ✅ Correctly Standardized |
| 0 (Easy) | 0.3% | 0 | ✅ Correctly Standardized |
| 0 (Easy) | 1.1% | 0 | ✅ Correctly Standardized |
| 0 (Easy) | 4.8% | 0 | ✅ Correctly Standardized |
| 0 (Easy) | 18.2% | 0 | ✅ Correctly Standardized |
| 1 (Hard) | 51.6% | 1 | ✅ Correctly Flagged (Marginal) |
| 1 (Hard) | 83.1% | 1 | ✅ Correctly Flagged |
| 1 (Hard) | 95.8% | 1 | ✅ Correctly Flagged |
| 1 (Hard) | 99.2% | 1 | ✅ Correctly Flagged |
| 1 (Hard) | 99.9% | 1 | ✅ Correctly Flagged |

---

## ⚙️ How to Initialize and Run the Workspace

Follow these steps to deploy this environment on your local development machine:

### Prerequisite System Environment
Ensure you have a modern Python 3 instance along with the core scientific computing stack installed:
```bash
pip install numpy pandas matplotlib jupyter

jupyter notebook

---

### Checking Your Work

Once saved, look at the top right corner of your VS Code window while editing the `README.md` file. You will see a small icon that looks like a split window with a magnifying glass (**Open Preview to the Side**). 

If you click that button, VS Code will show you exactly how the file will look when it gets uploaded to GitHub: all the hashes (`#`) turn into clean titles, the tables format beautifully, and the math symbols compile into professional math layout equations.

Your local coding project is now fully complete, beautifully documented, and ready for you to upload to your GitHub portfolio!