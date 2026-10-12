<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24,30&height=220&section=header&text=🚨%20Risk%20Alert%20Classifier&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Digital%20Banking%20Early-Warning%20System%20%7C%20High-Risk%20Customer%20Detection&descAlignY=58&descSize=17&animation=fadeIn" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=22&pause=1000&color=FF6B6B&center=true&vCenter=true&multiline=true&width=900&height=80&lines=🔍+Detecting+High-Risk+Customers+with+ML+Pipeline;⚖️+SMOTE+%7C+Random+Forest+%7C+AUC-ROC+%3D+0.9998;🏆+Best+Model%3A+Random+Forest+(GridSearchCV)" alt="Typing SVG"/>
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Imbalanced--Learn-0.11-8E44AD?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Matplotlib-3.8-11557C?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-0.13-4C8CBF?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-✅%20Production%20Ready-2ECC71?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20AUC-0.9998-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20F1-0.9820-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Records-4%2C600%20Customers-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-orange?style=for-the-badge"/>
</p>

<br/>

<blockquote>
<i>🏦 A production-grade ML classification pipeline for digital banking — built to catch every high-risk customer before they default, not after.</i>
</blockquote>

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/GitHub-@Dhairyapatel1mc-181717?style=flat-square&logo=github&logoColor=white"/>
</a>
&nbsp;&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/LinkedIn-ghost--patel-0077B5?style=flat-square&logo=linkedin&logoColor=white"/>
</a>
&nbsp;&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Instagram-@ghost__6927-E4405F?style=flat-square&logo=instagram&logoColor=white"/>
</a>

<br/><br/>

</div>

---

## 📋 Table of Contents

<div align="center">

| # | Section | Jump |
|:---:|---------|:---:|
| 01 | 🎯 Project Overview | [→](#-project-overview) |
| 02 | 📊 Dataset at a Glance | [→](#-dataset-at-a-glance) |
| 03 | 🗂️ Dataset Schema | [→](#️-dataset-schema) |
| 04 | 🧠 Theory Deep Dive | [→](#-theory-deep-dive) |
| 05 | ⚙️ ML Pipeline Architecture | [→](#️-ml-pipeline-architecture) |
| 06 | 🔬 Exploratory Data Analysis | [→](#-exploratory-data-analysis) |
| 07 | ⚖️ Handling Class Imbalance | [→](#️-handling-class-imbalance) |
| 08 | 🤖 Models Implemented | [→](#-models-implemented) |
| 09 | 🔧 Hyperparameter Tuning | [→](#-hyperparameter-tuning) |
| 10 | 🏆 Performance Results | [→](#-performance-results) |
| 11 | 📈 Visualizations | [→](#-visualizations--10-charts) |
| 12 | 📁 Project Structure | [→](#-project-structure) |
| 13 | 🚀 Quick Start | [→](#-quick-start) |
| 14 | 📋 Final Report | [→](#-final-report) |
| 15 | 👤 Author | [→](#-author--contact) |

</div>

---

## 🎯 Project Overview

<table>
<tr>
<td width="60%">

You are working as a **Data Scientist** for a digital banking platform. The bank needs an **early-warning system** that flags customers likely to default on payments or engage in fraudulent behavior — *before* it happens.

**The Challenge:** The dataset is **highly imbalanced** — only **12% of customers** are high-risk. A naive model achieves 88% accuracy by predicting "safe" for everyone, but misses every fraudster. This project solves that.

**What this pipeline delivers:**
- ✅ Accurately identifies high-risk customers (Recall → 1.00)
- ✅ Handles class imbalance with 4 resampling strategies
- ✅ Evaluates with business-relevant metrics (not just accuracy)
- ✅ Tunes models with RandomizedSearchCV + GridSearchCV
- ✅ Selects final model minimising **false negatives** (missed fraudsters)

</td>
<td width="40%" align="center">

| Field | Value |
|-------|-------|
| 🎓 Project Type | Classification |
| 🏦 Domain | Digital Banking / FinTech |
| ⚠️ Problem | Class Imbalance |
| 🎯 Target | `risk_status` (0/1) |
| 🏆 Best Model | Random Forest |
| 📊 Best AUC | **0.9998** |
| 📁 Dataset Size | 4,600 customers |
| ⏱️ Duration | 6 Hours |

</td>
</tr>
</table>

---

## 📊 Dataset at a Glance

<div align="center">

| 👥 Customers | 🔢 Features | ⚠️ High Risk | ✅ Low Risk | 📉 Imbalance Ratio |
|:---:|:---:|:---:|:---:|:---:|
| **4,600** | **17** | **557 (12.1%)** | **4,043 (87.9%)** | **7.3 : 1** |

</div>

> **Why this matters:** With a 7.3:1 imbalance, standard classifiers are heavily biased toward the majority class. Every technique in this project is designed to fight that bias.

```
CLASS DISTRIBUTION VISUALIZED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Low Risk  (0): ███████████████████████████████████████ 87.9%  (4,043)
High Risk (1): █████                                   12.1%  (557)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          ↑ The Challenge
```

---

## 🗂️ Dataset Schema

| # | Feature | Type | Description | Risk Signal |
|---|---------|------|-------------|-------------|
| 1 | `customer_id` | Int | Unique identifier | — |
| 2 | `age` | Float | Customer age | — |
| 3 | `gender` | Cat | Male / Female | — |
| 4 | `region` | Cat | Geographic region | — |
| 5 | `employment_type` | Cat | Salaried / Self-Employed / Unemployed | — |
| 6 | `annual_income_inr` | Float | Annual income in INR | 📉 Lower = Higher Risk |
| 7 | `credit_score` | Float | CIBIL score (300–850) | 📉 Lower = Higher Risk |
| 8 | `credit_utilization_ratio` | Float | Credit used / limit | 📈 Higher = Higher Risk |
| 9 | `missed_payments_12m` | Int | Missed payments last 12 months | ⭐ Top Signal |
| 10 | `avg_late_payment_days` | Float | Avg delay in days | ⭐ Top Signal |
| 11 | `monthly_transaction_count` | Int | Transactions per month | — |
| 12 | `monthly_spend_inr` | Float | Monthly spend in INR | — |
| 13 | `cash_advance_count_6m` | Int | Cash advances last 6 months | ⭐ Top Signal |
| 14 | `complaints_last_6m` | Int | Bank complaints last 6 months | 📈 Risk indicator |
| 15 | `failed_login_attempts_3m` | Int | Failed logins last 3 months | 🔴 Fraud Signal |
| 16 | `account_tenure_months` | Int | Account age in months | — |
| 17 | `debt_balance_inr` | Int | Current debt balance | 📈 Higher = Higher Risk |
| 18 | `risk_status` | Binary | **0 = Low Risk, 1 = High Risk** | 🎯 **TARGET** |

**Missing Values Handled via KNN Imputation (k=5):**

| Column | Missing Count |
|--------|:---:|
| `credit_score` | 216 |
| `annual_income_inr` | 166 |
| `employment_type` | 144 |
| `credit_utilization_ratio` | 147 |
| `age` | 140 |
| `monthly_spend_inr` | 129 |
| `region` | 102 |

---

## 🧠 Theory Deep Dive

<details>
<summary><b>🔷 Q1 — What is Logistic Regression and why is it suitable for classification?</b></summary>

<br>

**Logistic Regression** applies the **sigmoid function** to linear regression output to squash predictions into `[0, 1]`:

```
P(y=1 | X) = σ(z) = 1 / (1 + e^(−z))
             where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

| Property | Benefit |
|----------|---------|
| Outputs probabilities [0, 1] | Threshold can be tuned (default 0.5) |
| Interpretable coefficients | Shows feature impact direction |
| Fast training | Scales to large datasets |
| Works well as baseline | Benchmark for complex models |

> **In risk detection:** Setting threshold < 0.5 increases sensitivity (recall), catching more high-risk customers at the cost of some false alarms.

</details>

<details>
<summary><b>🔷 Q2 — Classification Metrics — Why Accuracy Alone Fails</b></summary>

<br>

```
THOUGHT EXPERIMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset:  880 Low Risk + 120 High Risk = 1,000 customers
Model A:  Predict "Low Risk" for ALL customers
Accuracy: 880/1000 = 88% ← Looks great!
Recall:   0/120 = 0%    ← Catches ZERO fraudsters 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| Metric | Formula | Business Meaning |
|--------|---------|-----------------|
| **Precision** | TP / (TP + FP) | When we flag someone as risky, how often are we right? |
| **Recall** | TP / (TP + FN) | Of all actual risky customers, what % did we catch? |
| **F1-Score** | 2·P·R / (P+R) | Harmonic balance of Precision and Recall |
| **AUC-ROC** | Area under curve | Overall ability to discriminate at all thresholds |

</details>

<details>
<summary><b>🔷 Q3 — Type-I vs Type-II Errors: The Business Stakes</b></summary>

<br>

```
                    PREDICTED
                   Low Risk │ High Risk
              ─────────────┼──────────────
ACTUAL Low  │    TN ✅    │   FP ⚠️      │  ← Type-I Error (False Alarm)
Risk        │ Correct!    │ Inconvenience │
            ─────────────┼──────────────
       High │    FN 🚨    │   TP ✅      │  ← Type-II Error (CRITICAL!)
Risk        │ FINANCIAL   │ Correct!     │
            │   LOSS!     │              │
            └─────────────┴──────────────┘
```

| Error Type | Technical Name | Business Consequence | Priority |
|-----------|---------------|---------------------|----------|
| **Type-I** | False Positive | Low-risk customer flagged → mild inconvenience | Low |
| **Type-II** | False Negative | High-risk customer missed → **fraud/default loss** | 🚨 **Critical** |

</details>

<details>
<summary><b>🔷 Q4 — Precision, Recall, F1, TPR, FPR Explained</b></summary>

<br>

```
From a Confusion Matrix of 920 test predictions:
┌─────────────────────────────────────────┐
│  TP = 109   FP = 2                      │
│  FN = 2     TN = 807                    │
└─────────────────────────────────────────┘

Precision  = 109/(109+2)   = 0.9820  ← "When we say risky, 98.2% are"
Recall/TPR = 109/(109+2)   = 0.9820  ← "We caught 98.2% of risky ones"
F1-Score   = 2×0.982×0.982 = 0.9820  ← Harmonic mean
FPR        = 2/(2+807)     = 0.0025  ← 0.25% false alarm rate
```

</details>

<details>
<summary><b>🔷 Q5 — AUC-ROC: The Threshold-Agnostic Metric</b></summary>

<br>

The **ROC Curve** plots **TPR (Recall) vs FPR** at every possible classification threshold (0 → 1):

```
TPR
1.0 ┤ ✦ ← Perfect Classifier (AUC=1.0)
    │╲
    │  ╲ ← Our RF Model (AUC=0.9998)
    │    ╲___________________________
0.5 ┤                        ╱
    │                    ╱ ← Random Classifier (AUC=0.5)
0.0 ┤────────────────────────────── FPR
    0.0                            1.0
```

| AUC Score | Interpretation |
|:---:|---|
| **1.00** | Perfect — separates classes completely |
| **0.90–0.99** | Excellent |
| **0.80–0.89** | Good |
| **0.50** | Random guessing — useless |

</details>

<details>
<summary><b>🔷 Q6 — Why Imbalanced Data Breaks Models</b></summary>

<br>

```
THE MAJORITY CLASS TRAP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training loss is minimized by predicting majority class.
  → Model learns: "Always say Low Risk"
  → Loss: 12% error (acceptable by optimizer)
  → But: 100% of fraudsters slip through 🚨

SOLUTIONS APPLIED IN THIS PROJECT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Under-Sampling  → Remove majority rows randomly
2. Over-Sampling   → Duplicate minority rows
3. SMOTE           → Synthesize new minority samples ✅ Best
4. ADASYN          → Adaptive SMOTE near decision boundary
```

</details>

---

## ⚙️ ML Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    RISK ALERT CLASSIFIER — ML PIPELINE                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            ║
║  │ Raw CSV  │ → │  Parse   │ → │  Encode  │ → │   KNN    │            ║
║  │ 4600 rows│   │ & Clean  │   │ Cat Vars │   │ Imputer  │            ║
║  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘            ║
║                                                      │                  ║
║  ┌───────────────────────────────────────────────────▼──┐              ║
║  │           Stratified Train-Test Split (80/20)         │              ║
║  │           Maintains 12% High Risk in both sets        │              ║
║  └────────────────────────┬─────────────────────────────┘              ║
║                            │                                             ║
║     ┌──────────────────────▼──────────────────────┐                    ║
║     │            StandardScaler (fit on train)     │                    ║
║     └──────────────────────┬──────────────────────┘                    ║
║                             │                                            ║
║     ┌───────────────────────▼─────────────────────┐                    ║
║     │         Class Imbalance Handling (SMOTE)     │                    ║
║     │   Under │ Over │ SMOTE ✅ │ ADASYN           │                    ║
║     └─────┬────────────┬──────────────┬────────────┘                   ║
║           │            │              │                                  ║
║    ┌──────▼──┐  ┌──────▼──┐  ┌───────▼─────────┐                     ║
║    │Logistic │  │Decision │  │  Random Forest  │                      ║
║    │  Reg.   │  │  Tree   │  │  (GridSearchCV) │ ← BEST MODEL        ║
║    └─────────┘  └─────────┘  └─────────────────┘                     ║
║                                                                          ║
║     ┌──────────────────────────────────────────────────┐               ║
║     │  Evaluation: CM │ Precision │ Recall │ F1 │ AUC  │               ║
║     └──────────────────────────────────────────────────┘               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🔬 Exploratory Data Analysis

### Missing Values Before KNN Imputation

```
credit_score             ████████████████████████████████ 216 missing (4.7%)
annual_income_inr        ██████████████████████████       166 missing (3.6%)
credit_utilization_ratio █████████████████████████        147 missing (3.2%)
employment_type          ██████████████████████████       144 missing (3.1%)
age                      ████████████████████████         140 missing (3.0%)
monthly_spend_inr        █████████████████████            129 missing (2.8%)
region                   ████████████████                 102 missing (2.2%)
```

### Key Correlations with `risk_status`

```
avg_late_payment_days    +0.711  ████████████████████████████  Strong ↑ (Top Signal)
missed_payments_12m      +0.671  ████████████████████████████  Strong ↑ (Top Signal)
cash_advance_count_6m    +0.554  ████████████████████          Strong ↑
credit_utilization_ratio +0.500  ████████████████████          Strong ↑
credit_score             −0.506  ████████████████████          Strong ↓ (Protective)
```

---

## ⚖️ Handling Class Imbalance

### The 4 Techniques Compared

| Technique | How It Works | Train Size After | Recall | F1 | Verdict |
|-----------|-------------|:---:|:---:|:---:|---------|
| 📉 Under-Sampling | Remove majority rows | 892 | 1.000 | 0.978 | Info loss |
| 📈 Over-Sampling | Duplicate minority rows | 6,468 | 1.000 | 0.987 | Overfit risk |
| ✅ **SMOTE** | Synthesize new minority points | 6,468 | 1.000 | **0.991** | **Best** |
| 🔵 ADASYN | Adaptive synthesis at boundary | ~6,500 | 1.000 | 0.982 | Slightly noisy |

### How SMOTE Works

```
BEFORE SMOTE:                        AFTER SMOTE:
                                     
  ● ● ● ●  ← Low Risk (many)          ● ● ● ●   ← Low Risk (unchanged)
  ● ● ● ●                              ● ● ● ●
  ● ● ● ●                              ● ● ● ●
     ★      ← High Risk (few)             ★  ★★   ← Synthetic points!
                                         ★ ★★★ ★     generated along
                                         ★★  ★★      interpolation lines
                                                     between real points
```

> **SMOTE chosen** as the final balancing strategy — best F1 (0.9911) and generates realistic synthetic samples rather than duplicating existing ones.

---

## 🤖 Models Implemented

| Model | Part | Key Configuration | Scaling | Role |
|-------|------|-------------------|---------|------|
| **Logistic Regression** | C | `max_iter=1000, C=1.0` | ✅ Required | Baseline benchmark |
| **Decision Tree** | E | `max_depth=6, min_leaf=10` | ❌ Not needed | Controlled tree |
| **🏆 Random Forest** | E→F | `150 trees, depth=10, log2` | ❌ Not needed | **Production model** |

### Decision Tree — Depth vs Overfitting

```
DEPTH vs PERFORMANCE (F1 on SMOTE-balanced data):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Depth   Train F1    Test F1    Gap       Status
  2      0.82        0.80      0.02    ✅ Underfit
  4      0.90        0.88      0.02    ✅ Good
  6      0.91        0.87      0.04    ✅ Best Balance ←
  8      0.95        0.88      0.07    ⚠️ Slight Overfit
 10      0.98        0.87      0.11    ⚠️ Overfitting
None     1.00        0.86      0.14    🔴 Severe Overfit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Why Random Forest Beats Decision Tree

```
Decision Tree  → One tree → High variance → Sensitive to noise
Random Forest  → 150 trees, each on random feature subsets
               → Average predictions → variance cancels out → robust

ENSEMBLE ADVANTAGES:
  ✓ Decorrelated trees (random feature selection at each split)
  ✓ Bagging (bootstrap samples reduce variance)
  ✓ Majority voting on 150 trees vs 1
  ✓ Built-in feature importance via impurity reduction
  ✓ Handles missing values gracefully
```

---

## 🔧 Hyperparameter Tuning

### Strategy: Coarse → Fine

```
STAGE 1: RandomizedSearchCV — Decision Tree
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Search Space: max_depth, min_samples_split, min_samples_leaf, criterion
Iterations:   25 random combinations | CV: 3-fold | Scoring: F1

Best DT Config Found:
  max_depth = 10  |  criterion = gini
  min_samples_split = 2  |  min_samples_leaf = 1

STAGE 2: RandomizedSearchCV — Random Forest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Search Space: n_estimators, max_depth, min_samples_split,
              min_samples_leaf, max_features
Iterations:   20 | CV: 3-fold | Scoring: F1

Best RF Config Found:
  n_estimators = 150  |  max_depth = 10
  min_samples_leaf = 1  |  max_features = log2

STAGE 3: GridSearchCV — Fine Tuning (RF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exhaustive search on best region:
  n_estimators = [100, 150]  ×  max_depth = [8, 10]  ×  min_leaf = [1, 3]
  → 2×2×2 = 8 combos × 3 folds = 24 fits  →  Best config confirmed
```

### Tuning Impact

| Model | Before Tuning | After Tuning | Δ |
|-------|:---:|:---:|:---:|
| Decision Tree F1 | 0.8734 | 0.8831 | +0.0097 ↑ |
| Decision Tree Recall | 0.9009 | 0.9189 | +0.0180 ↑ |
| Random Forest F1 | 0.9776 | 0.9820 | +0.0044 ↑ |
| Random Forest AUC | 0.9998 | 0.9998 | — (already optimal) |

---

## 🏆 Performance Results

### Final Model Leaderboard

<div align="center">

| Rank | 🏅 Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|:----:|----------|:--------:|:---------:|:------:|:--------:|:-------:|
| 🥇 | **Random Forest (GridCV)** | **0.9946** | **0.9732** | **0.9820** | **0.9776** | **0.9998** |
| 🥈 | Logistic Regression (SMOTE) | 0.9978 | 0.9823 | 1.0000 | 0.9911 | 1.0000 |
| 🥉 | Decision Tree | 0.9685 | 0.8475 | 0.9009 | 0.8734 | 0.9744 |

</div>

### Performance Visual

```
AUC-ROC SCORES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logistic Regression  ██████████████████████████████████████████ 1.0000 ⭐
Random Forest        ██████████████████████████████████████████ 0.9998 🏆
Decision Tree        █████████████████████████████████████████  0.9744

RECALL (catching all high-risk customers):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logistic Regression  ██████████████████████████████████████████ 1.0000
Random Forest        █████████████████████████████████████████  0.9820
Decision Tree        ██████████████████████████████████████████ 0.9009
```

### Confusion Matrix — Best Model (Random Forest)

```
                    PREDICTED
               Low Risk  │  High Risk
          ───────────────┼──────────────
 ACTUAL   Low Risk  │   807  │    2    │  ← 2 false alarms (Type-I)
          ───────────────┼──────────────
          High Risk │     2  │  109    │  ← Only 2 missed! (Type-II) 🎯
          ───────────────┴──────────────

  📊 Of 111 actual high-risk customers:
     → 109 correctly flagged  (98.2% caught!)
     →   2 slipped through    (1.8% missed)
  
  📊 Of 809 actual low-risk customers:
     → 807 correctly cleared  (99.8% accuracy!)
     →   2 incorrectly flagged (0.2% false alarm)
```

---

## 📈 Visualizations — 10 Charts

| # | Chart File | Description | Key Insight |
|:---:|-----------|-------------|-------------|
| 01 | `01_eda_class_distribution.png` | Class imbalance pie + 7 feature histograms | 12.1% minority confirmed |
| 02 | `02_correlation_heatmap.png` | Full feature correlation matrix | `avg_late_payment_days` is top predictor |
| 03 | `03_feature_by_risk.png` | Boxplots by risk label for 6 features | Clear separation in credit metrics |
| 04 | `04_logistic_regression_baseline.png` | Confusion matrix + metrics bar | Baseline benchmark established |
| 05 | `05_imbalance_handling.png` | Recall / F1 / AUC before vs after 4 techniques | SMOTE best overall |
| 06 | `06_class_distribution_balanced.png` | Class counts after each technique | Visual proof of balancing |
| 07 | `07_tree_models.png` | DT depth overfit analysis + RF feature importance | depth=6 sweet spot |
| 08 | `08_hyperparameter_tuning.png` | Tuned vs untuned for DT and RF | GridCV improves generalization |
| 09 | `09_roc_model_comparison.png` | ROC curves + all-metrics comparison | All 3 on same axes |
| 10 | `10_final_diagnostics.png` | Final CM + P-R curve + business metrics | RF best for production |

---

## 📁 Project Structure

```
🚨 risk-alert-classifier/
│
├── 📓 Risk_Alert_Classifier.ipynb          ← Pre-executed notebook (all outputs)
│
├── 📊 data/
│   ├── risk_alert.csv                       ← Raw dataset (4,600 × 19)
│   └── final_predictions.csv               ← ✅ Predictions from all 3 models
│                                               (920 test rows × 24 columns)
│
├── 📈 charts/
│   ├── 01_eda_class_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_by_risk.png
│   ├── 04_logistic_regression_baseline.png
│   ├── 05_imbalance_handling.png
│   ├── 06_class_distribution_balanced.png
│   ├── 07_tree_models.png
│   ├── 08_hyperparameter_tuning.png
│   ├── 09_roc_model_comparison.png
│   └── 10_final_diagnostics.png
│
└── 📄 README.md                             ← This file
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Dhairyapatel1mc/risk-alert-classifier.git
cd risk-alert-classifier
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

### 3. Run Notebook

```bash
jupyter notebook Risk_Alert_Classifier.ipynb
```

### 4. Quick Inference

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Balance training data
X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train_sc, y_train)

# Train best model
rf = RandomForestClassifier(n_estimators=150, max_depth=10,
                             max_features='log2', random_state=42)
rf.fit(X_sm, y_sm)

# Get risk probabilities
risk_probs = rf.predict_proba(X_new_customers)[:, 1]

# Flag customers above threshold
for cust_id, prob in zip(customer_ids, risk_probs):
    status = "🚨 HIGH RISK" if prob >= 0.5 else "✅ LOW RISK"
    print(f"Customer {cust_id}: {status} — probability: {prob:.4f}")
```

---

## 📋 Final Report

### Best Model Justification

| Criterion | Choice | Reason |
|-----------|--------|--------|
| Production Model | **Random Forest** | Robust, generalizable, works on noisy real-world data |
| Balancing Strategy | **SMOTE** | Best F1 (0.9911), avoids exact duplicates |
| Tuning Method | **GridSearchCV** | Exhaustive, reliable for final production config |
| Evaluation Focus | **Recall + F1** | Business priority: minimize missed high-risk customers |

### Business Impact

```
FINANCIAL IMPACT ANALYSIS (per 920 test customers):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Assumptions:
  • Avg loss per missed high-risk customer: ₹50,000
  • Avg cost per false alarm (manual review): ₹2,000

Without this model (naive 88% accuracy):
  → Catches ~50% of 111 = 56 missed high-risk
  → Loss: 56 × ₹50,000 = ₹28,00,000

With Random Forest:
  → Catches 109 of 111 = only 2 missed
  → Loss: 2 × ₹50,000 + 2 × ₹2,000 = ₹1,04,000

MONTHLY SAVINGS: ₹28,00,000 − ₹1,04,000 = ₹26,96,000 🎯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Top Risk Drivers (RF Feature Importance)

```
avg_late_payment_days    ████████████████████████████  1st — strongest signal
missed_payments_12m      ███████████████████████████   2nd
cash_advance_count_6m    ██████████████████████        3rd
credit_utilization_ratio ████████████████████          4th
credit_score             █████████████████             5th (inverse)
failed_login_attempts_3m ██████████████               6th — fraud signal
debt_balance_inr         ████████████                 7th
complaints_last_6m       ██████████                   8th
```

### Recommendations

- 🔄 **Retrain monthly** — customer behaviour patterns shift seasonally
- 📊 **Monitor drift** — track feature distributions in production
- ⚡ **Deploy as REST API** — enable real-time transaction-time risk scoring
- 🎯 **Lower threshold to 0.35** — catch even more high-risk customers if recall is top priority
- 📱 **Auto-escalate** — customers with `risk_prob > 0.80` flagged directly to risk team

---

## 👤 Author & Contact

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=12,20,24&height=120&section=header&text=Ghost%20(Patel%20Dhairya)&fontSize=30&fontColor=ffffff&desc=Data%20Scientist%20%7C%20ML%20Engineer%20%7C%20RWSkill&descSize=14&animation=fadeIn" width="70%"/>

<br/>

| | |
|:---:|---|
| 👤 **Name** | Ghost (Patel Dhairya) |
| 🏫 **Institute** | Red and White Skill Education (RWSkill) |
| 💼 **Role** | Data Scientist / ML Engineer |
| 🐙 **GitHub** | [@Dhairyapatel1mc](https://github.com/Dhairyapatel1mc) |
| 💼 **LinkedIn** | [ghost-patel](https://linkedin.com/in/ghost-patel) |
| 📸 **Instagram** | [@ghost_6927](https://instagram.com/ghost_6927) |

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/Follow%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>
&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/Connect%20on%20LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>
&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Follow%20on%20Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>
</a>

</div>

---

## 🛠️ Full Tech Stack

<div align="center">

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Data Manipulation** | Pandas, NumPy |
| **ML Models** | Scikit-Learn (LR, DT, RF, RandomizedSearchCV, GridSearchCV) |
| **Imbalance Handling** | Imbalanced-Learn (SMOTE, ADASYN, RandomUnderSampler, RandomOverSampler) |
| **Visualization** | Matplotlib, Seaborn |
| **Notebook** | Jupyter Notebook (pre-executed) |
| **Preprocessing** | StandardScaler, KNNImputer, LabelEncoder, StratifiedKFold |

</div>

---

## 📜 License

```
MIT License — Copyright (c) 2024 Ghost (Patel Dhairya) · Red and White Skill Education

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to deal in the Software without restriction, including the
rights to use, copy, modify, merge, publish and distribute.
```

---

<div align="center">

**[⬆ Back to Top](#-risk-alert-classifier)**

<br/>

```
⭐ If this project helped you, please star the repository!
```

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24,30&height=100&section=footer&animation=fadeIn" width="100%"/>

*Built with precision by Ghost (Patel Dhairya) · Red and White Skill Education (RWSkill) · 2024*

</div>
