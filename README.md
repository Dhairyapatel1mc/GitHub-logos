<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,18,24&height=220&section=header&text=📨%20Message%20Intelligence%20System&fontSize=44&fontColor=ffffff&fontAlignY=38&desc=Spam%20Detection%20%7C%20KNN%20·%20SVM%20·%20Naive%20Bayes%20·%20Probability%20Theory&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=20&pause=1000&color=E91E8C&center=true&vCenter=true&multiline=true&width=860&height=80&lines=📩+Classifying+Spam+vs+Legitimate+Messages;🔢+KNN+%7C+SVM+%7C+Naive+Bayes+%7C+AUC+%3D+1.0000;🧠+Bayes'+Theorem+·+Margin+·+Distance+Classifiers" alt="Typing SVG"/>
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-3.8-11557C?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-✅%20Completed-2ECC71?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20AUC-1.0000-E91E8C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Messages-500%20Total-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-orange?style=for-the-badge"/>
</p>

<br/>
<blockquote>
<i>📡 A communication security system that identifies whether incoming digital messages are Spam or Legitimate — combining probability theory with distance-based, margin-based, and probabilistic classifiers.</i>
</blockquote>
<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/GitHub-@Dhairyapatel1mc-181717?style=flat-square&logo=github"/>
</a>
&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/LinkedIn-ghost--patel-0077B5?style=flat-square&logo=linkedin"/>
</a>
&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Instagram-@ghost__6927-E4405F?style=flat-square&logo=instagram"/>
</a>

<br/><br/>

</div>

---

## 📋 Table of Contents

<div align="center">

| # | Section |
|:---:|---------|
| 01 | [🎯 Project Overview](#-project-overview) |
| 02 | [🔄 Data Pipeline — Animated](#-data-pipeline--from-raw-to-ready) |
| 03 | [📊 Dataset at a Glance](#-dataset-at-a-glance) |
| 04 | [🗂️ Dataset Schema](#️-dataset-schema) |
| 05 | [🧠 Theory Cards — Animated](#-theory--animated-concept-cards) |
| 06 | [📏 KNN — Animated Visual](#-knn--animated-working-principle) |
| 07 | [⚡ SVM — Animated Visual](#-svm--animated-margin--kernels) |
| 08 | [📊 Naive Bayes — Animated](#-naive-bayes--animated-bayes-theorem) |
| 09 | [🏆 Classifier Comparison — Animated](#-classifier-comparison--animated) |
| 10 | [🏆 Performance Results](#-performance-results) |
| 11 | [📈 Visualizations](#-visualizations--10-charts) |
| 12 | [📁 Project Structure](#-project-structure) |
| 13 | [🚀 Quick Start](#-quick-start) |
| 14 | [📋 Final Report](#-final-report) |
| 15 | [👤 Author](#-author--contact) |

</div>

---

## 🎯 Project Overview

<table>
<tr>
<td width="55%">

You are working as a **Data Scientist** for a **communication security company**. The company wants to build a **Message Intelligence System** that automatically classifies user messages as:

- `0` → **Legitimate Message** ✅
- `1` → **Spam Message** 🚨

The dataset contains **message-related features** extracted from text and user behavior signals. Your task is to **build and compare multiple classification models** and explain their performance using **probability concepts**.

**What this pipeline delivers:**
- ✅ KNN — Distance-based classifier (k=5)
- ✅ SVM — Linear, RBF, Polynomial kernels
- ✅ Naive Bayes — Bayes' Theorem + manual calculation
- ✅ ROC, Precision-Recall, Confusion Matrix analysis

</td>
<td width="45%" align="center">

| Field | Value |
|-------|-------|
| 🎓 Type | Classification |
| 🏢 Domain | Communication Security |
| 📩 Problem | Spam Detection |
| 🎯 Target | `spam_label` (0/1) |
| 🏆 Best AUC | **1.0000** |
| 📁 Messages | 500 total |
| 📐 Features | 12 |
| ⏱️ Duration | 6 Hours |

</td>
</tr>
</table>

---

## 🔄 Data Pipeline — From Raw to Ready

> 🎬 **Animated SVG** — Watch the pipeline build stage by stage when you open this README on GitHub!

![Data Pipeline](pipeline.svg)

| Stage | Action | Input | Output |
|:---:|--------|-------|--------|
| **01 / SOURCE** | Inspect structure | Raw CSV 500×16 | Shape, dtypes understood |
| **02 / PROFILE** | Quality check | Raw features | Missing=0, Duplicates=0 ✅ |
| **03 / CLEAN** | Preprocess & scale | Raw values | StandardScaler applied |
| **04 / OUTPUT** | Analysis-ready | Scaled features | Train(400) + Test(100) |

---

## 📊 Dataset at a Glance

<div align="center">

| 📩 Total | ✅ Legitimate | 🚨 Spam | 📐 Features | 🔀 Train | 🧪 Test |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **500** | **300 (60%)** | **200 (40%)** | **12** | **400** | **100** |

</div>

```
CLASS DISTRIBUTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Legitimate (0): ████████████████████████████████  60%  (300)
Spam       (1): ████████████████████              40%  (200)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🗂️ Dataset Schema

| # | Feature | Type | Description | Signal |
|---|---------|------|-------------|--------|
| 1 | `message_id` | Int | Unique ID | — |
| 2 | `message_text` | String | Raw message content | — |
| 3 | `message_length` | Int | Character count | — |
| 4 | `word_count` | Int | Word count | — |
| 5 | `num_urls` | Int | URLs in message | 📈 Spam |
| 6 | `num_digits` | Int | Digit count | 📈 Spam |
| 7 | `num_special_chars` | Int | Special char count | 📈 Spam |
| 8 | `spam_keyword_score` | Float | Spam keyword frequency | ⭐ Top Signal |
| 9 | `legit_keyword_score` | Float | Legit keyword score | ⭐ Top Signal |
| 10 | `sender_activity_score` | Float | Sender's activity | 📉 Spam |
| 11 | `sender_account_age_days` | Int | Account age (days) | ⭐ Top Signal |
| 12 | `messages_sent_last_24h` | Int | Messages last 24h | 📈 Spam |
| 13 | `hour_of_day` | Int | Send hour | — |
| 14 | `day_of_week` | Int | Send day | — |
| 15 | `timestamp` | DateTime | Message timestamp | — |
| 16 | `spam_label` | Binary | **0=Legit, 1=Spam** | 🎯 **TARGET** |

---

## 🧠 Theory — Animated Concept Cards

> 🎬 All cards below are **animated SVGs** — elements slide in, bars grow, and diagrams build themselves on GitHub!

---

### 📏 KNN — Animated Working Principle

> Watch the data points appear, the query point drop in, the radar ring expand, and distance lines draw to the 5 nearest neighbors!

![KNN Classifier](knn.svg)

<details>
<summary><b>📖 KNN Theory Notes</b></summary>

<br>

**K-Nearest Neighbors** is a lazy, non-parametric, distance-based classifier:

```
ALGORITHM:
  1. Store ALL training points (no model building)
  2. New message arrives → compute distance to every point
  3. Find K nearest neighbors
  4. Majority class vote → prediction

DISTANCE METRICS:
  Euclidean:  d = √Σ(xᵢ−yᵢ)²      → chosen (k=5, F1=1.000)
  Manhattan:  d = Σ|xᵢ−yᵢ|        → equivalent performance
  Minkowski:  d = (Σ|xᵢ−yᵢ|^p)^(1/p) → generalization

K SELECTION:
  k=1  → overfit (memorizes noise)
  k=5  → best balance ✅ CHOSEN
  k=21 → underfit (over-smoothed)
```

</details>

---

### ⚡ SVM — Animated Margin & Kernels

> Watch spam and legit points appear, the hyperplane draw itself, support vectors get circled, and all three kernel cards slide in!

![SVM Classifier](svm.svg)

<details>
<summary><b>📖 SVM Theory Notes</b></summary>

<br>

**Support Vector Machine** finds the maximum margin hyperplane:

```
CORE IDEA:
  Maximize margin = 2 / ||w||
  Subject to: yᵢ(w·xᵢ + b) ≥ 1  for all i

KERNELS:
  Linear:     K(x,y) = x·y            → F1=1.000 ✅
  RBF:        K(x,y) = exp(-γ||x-y||²) → F1=1.000 ✅ CHOSEN
  Polynomial: K(x,y) = (x·y+c)^d      → F1=0.974

SUPPORT VECTORS:
  Points that lie ON the margin boundary
  Only these define the hyperplane (sparse solution)
  All other points → irrelevant for prediction
```

</details>

---

### 📊 Naive Bayes — Animated Bayes Theorem

> Watch the four formula blocks slide in, the step-by-step calculation reveal, and the final SPAM result flash!

![Naive Bayes & Bayes Theorem](bayes.svg)

<details>
<summary><b>📖 Naive Bayes Theory Notes</b></summary>

<br>

**Bayes' Theorem** applied to spam classification:

```
P(Spam|Features) = P(Features|Spam) × P(Spam) / P(Features)
                    ─────────────────   ──────────  ────────────
                      Likelihood          Prior       Evidence

MANUAL CALCULATION — Sample message (spam_keyword_score=+1.321):
  Prior:      P(Spam) = 200/500 = 0.4000
  Likelihood: P(x|Spam)  = N(1.321; μ=+1.112, σ=0.714) = 0.538
              P(x|Legit) = N(1.321; μ=−0.741, σ=0.621) ≈ 0.000
  Posterior:  P(Spam|x) = 0.538×0.40 / norm = 1.0000 → SPAM 🚨

ASSUMPTIONS:
  ✓ Feature independence (Naive)
  ✓ Gaussian distribution per class (GaussianNB)
  ✓ Parameters estimated by MLE on training data
```

</details>

---

## 🏆 Classifier Comparison — Animated

> Watch all three classifier panels slide in, Gaussian PDFs draw for Naive Bayes, the SVM margin appear, KNN neighbors cluster — then the performance bars grow from left to right!

![Classifier Comparison](classifiers.svg)

---

## 🏆 Performance Results

### Final Model Leaderboard

<div align="center">

| Rank | 🏅 Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|:----:|----------|:--------:|:---------:|:------:|:--------:|:-------:|
| 🥇 | **KNN (k=5)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 🥇 | **SVM Linear** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 🥇 | **SVM RBF** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 🥇 | **Naive Bayes** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 4 | SVM Polynomial | 0.9800 | 1.0000 | 0.9500 | 0.9744 | 1.0000 |

</div>

```
AUC-ROC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNN (k=5)    ██████████████████████████████████ 1.0000
SVM Linear   ██████████████████████████████████ 1.0000
SVM RBF      ██████████████████████████████████ 1.0000
Naive Bayes  ██████████████████████████████████ 1.0000
SVM Poly     █████████████████████████████████  1.0000 (AUC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Best Model for Each Use Case

| Use Case | Best Model | Reason |
|----------|-----------|--------|
| 🎯 High Precision | SVM Linear | Interpretable, max margin |
| 🔍 High Recall | All models | 100% recall achieved |
| 📊 Probability scores | Naive Bayes | P(Spam\|message) directly |
| ⚡ Real-time prod | SVM RBF | Fast inference after training |
| 🔎 Explainability | Naive Bayes | Bayes theorem traceable |

---

## 📈 Visualizations — 10 Charts

| # | Chart | Key Insight |
|:---:|-------|-------------|
| 01 | EDA Distributions | Clear bimodal separation in spam_keyword_score |
| 02 | Correlation Heatmap | spam_kw & legit_kw strongly anti-correlated |
| 03 | Feature by Class | Near-zero overlap → linearly separable |
| 04 | KNN Analysis | k=5 optimal across all distance metrics |
| 05 | SVM Analysis | Linear & RBF achieve perfect separation |
| 06 | Naive Bayes | Gaussian PDFs clearly separated by class |
| 07 | ROC Curves | All models hug top-left corner |
| 08 | Precision-Recall | Perfect precision at all recall levels |
| 09 | Bayes Probability | Manual calc matches model exactly |
| 10 | Final Diagnostics | Classifier taxonomy summary |

---

## 📁 Project Structure

```
📨 message-intelligence-system/
│
├── 📓 Message_Intelligence_System.ipynb     ← Pre-executed notebook
│
├── 📊 data/
│   ├── messages.csv                          ← Dataset (500 × 16)
│   └── final_predictions.csv                ← Predictions all 4 models
│
├── 📈 charts/
│   ├── 01_eda_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_feature_by_class.png
│   ├── 04_knn_analysis.png
│   ├── 05_svm_analysis.png
│   ├── 06_naive_bayes.png
│   ├── 07_roc_comparison.png
│   ├── 08_precision_recall.png
│   ├── 09_bayes_probability.png
│   └── 10_final_diagnostics.png
│
├── 🎨 (SVG Animations — same folder as README)
│   ├── pipeline.svg       ← 4-stage data pipeline animation
│   ├── knn.svg            ← KNN working principle animation
│   ├── svm.svg            ← SVM margin & kernels animation
│   ├── bayes.svg          ← Bayes theorem step-by-step animation
│   └── classifiers.svg    ← 3-classifier comparison animation
│
└── 📄 README.md                              ← This file
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Dhairyapatel1mc/message-intelligence-system.git
cd message-intelligence-system

# Install
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter

# Run
jupyter notebook Message_Intelligence_System.ipynb
```

```python
# Quick inference
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

svm = SVC(kernel='rbf', probability=True).fit(X_train_sc, y_train)
probs = svm.predict_proba(X_test_sc)[:, 1]

for msg_id, prob in zip(ids, probs):
    label = "🚨 SPAM" if prob >= 0.5 else "✅ LEGIT"
    print(f"Message {msg_id}: {label} | P(spam)={prob:.4f}")
```

---

## 📋 Final Report

```
╔══════════════════════════════════════════════════════════════╗
║         STRENGTHS & WEAKNESSES — ALL CLASSIFIERS             ║
╠══════════════════╦═══════════════════════════════════════════╣
║  KNN (k=5)       ║  ✅ Intuitive, adapts to local geometry   ║
║                  ║  ❌ Slow inference O(n·d), scaling needed  ║
╠══════════════════╬═══════════════════════════════════════════╣
║  SVM Linear      ║  ✅ Fast, interpretable weights            ║
║                  ║  ❌ Poor on non-linear without kernel      ║
╠══════════════════╬═══════════════════════════════════════════╣
║  SVM RBF         ║  ✅ Handles non-linearity, robust          ║
║                  ║  ❌ Slow training, needs tuning            ║
╠══════════════════╬═══════════════════════════════════════════╣
║  Naive Bayes     ║  ✅ Probabilistic, fast, interpretable     ║
║                  ║  ❌ Independence assumption rarely true     ║
╚══════════════════╩═══════════════════════════════════════════╝
```

**Top Features:** `spam_keyword_score` · `legit_keyword_score` · `sender_account_age_days`

**Business Recommendation:**
- 🚀 Deploy **SVM RBF** as primary real-time classifier
- 📊 Use **Naive Bayes** for probability-based risk scoring
- 🔄 Retrain weekly as spam patterns evolve

---

## 👤 Author & Contact

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=6,14,18&height=120&section=header&text=Ghost%20(Patel%20Dhairya)&fontSize=28&fontColor=ffffff&desc=Data%20Scientist%20%7C%20ML%20Engineer%20%7C%20RWSkill&descSize=13&animation=fadeIn" width="65%"/>

<br/>

| | |
|:---:|---|
| 👤 | Ghost (Patel Dhairya) |
| 🏫 | Red and White Skill Education (RWSkill) |
| 🐙 | [@Dhairyapatel1mc](https://github.com/Dhairyapatel1mc) |
| 💼 | [ghost-patel](https://linkedin.com/in/ghost-patel) |
| 📸 | [@ghost_6927](https://instagram.com/ghost_6927) |

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

## 📜 License

```
MIT License — Copyright (c) 2024 Ghost (Patel Dhairya) · Red and White Skill Education
```

---

<div align="center">

**[⬆ Back to Top](#-message-intelligence-system)**

<br/>

```
⭐ If this helped you, please star the repo!
```

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,18,24&height=100&section=footer&animation=fadeIn" width="100%"/>

*Built with ❤️ by Ghost (Patel Dhairya) · RWSkill · 2024*

</div>
