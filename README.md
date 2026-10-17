
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,18,24&height=220&section=header&text=📨%20Message%20Intelligence%20System&fontSize=44&fontColor=ffffff&fontAlignY=38&desc=Spam%20Detection%20%7C%20KNN%20·%20SVM%20·%20Naive%20Bayes%20·%20Probability%20Theory&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=20&pause=1000&color=E91E8C&center=true&vCenter=true&multiline=true&width=900&height=80&lines=📩+Classifying+Spam+vs+Legitimate+Messages+with+ML;🔢+KNN+%7C+SVM+%7C+Naive+Bayes+%7C+AUC+%3D+1.0000;🧠+Bayes'+Theorem+·+Distance+·+Margin+Classifiers" alt="Typing SVG"/>
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
| 02 | [🔄 Data Pipeline](#-data-pipeline--from-raw-to-ready) |
| 03 | [📊 Dataset at a Glance](#-dataset-at-a-glance) |
| 04 | [🗂️ Dataset Schema](#️-dataset-schema) |
| 05 | [🧠 Theory & Concept Cards](#-theory--concept-cards) |
| 06 | [⚙️ ML Pipeline Architecture](#️-ml-pipeline-architecture) |
| 07 | [📏 KNN — Distance-Based Classifier](#-knn--distance-based-classifier) |
| 08 | [⚡ SVM — Margin-Based Classifier](#-svm--margin-based-classifier) |
| 09 | [📊 Naive Bayes — Probabilistic Classifier](#-naive-bayes--probabilistic-classifier) |
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

> **Inspired by the Data Workflow diagram — 4-stage pipeline from raw data to analysis-ready dataset.**

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                    PART B / DATA WORKFLOW — Dataset Understanding & Preparation          ║
║                         FROM RAW DATA TO ANALYSIS-READY DATA                            ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  01 / SOURCE          02 / PROFILE         03 / CLEAN          04 / OUTPUT              ║
║  ┌───────────┐        ┌───────────┐        ┌───────────┐       ┌───────────┐            ║
║  │           │        │ ░░░░░░░░░ │        │ RAW → CLN │       │  ✅ DATA  │            ║
║  │  Raw CSV  │  →→→   │ Quality   │  →→→   │ Fix Issues│  →→→  │  READY   │            ║
║  │  500 rows │        │ Profile   │        │ Validate  │       │ Validated │            ║
║  │  16 cols  │        │           │        │           │       │           │            ║
║  └───────────┘        └───────────┘        └───────────┘       └───────────┘            ║
║                                                                                          ║
║  Inspect structure    Find quality issues  Fix inconsistencies  Ready for analysis       ║
║  Rows, cols, types    Profile & validate   Duplicates & nulls   Structured & reliable   ║
║  Original input       Quality check        Clean records         Final dataset           ║
║                                                                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║  DATA QUALITY CONTROL — Preparation Checklist                                           ║
║  ✓ Column types checked      ✓ Missing values reviewed                                  ║
║  ✓ Duplicate records checked ✓ Dataset validated                                        ║
║                         ► READY FOR ANALYSIS — Reliable · Structured · Usable          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
```

### Stage-by-Stage Breakdown

| Stage | Action | Details | Output |
|:---:|--------|---------|--------|
| **01 SOURCE** | Inspect Raw Data | Shape, dtypes, .info(), .describe() | Structure understood |
| **02 PROFILE** | Quality Check | Missing values: 0 ✅, Duplicates: 0 ✅ | Data profiled |
| **03 CLEAN** | Preprocess | Label encoding, StandardScaler, stratified split | Clean dataset |
| **04 OUTPUT** | Ready for ML | X_train (400), X_test (100), scaled features | Analysis-ready |

---

## 📊 Dataset at a Glance

<div align="center">

| 📩 Total Messages | ✅ Legitimate | 🚨 Spam | 📐 Features | 🔀 Train Split | 🧪 Test Split |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **500** | **300 (60%)** | **200 (40%)** | **12** | **400 (80%)** | **100 (20%)** |

</div>

```
MESSAGE CLASS DISTRIBUTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Legitimate (0): ████████████████████████████████  60%  (300)
Spam       (1): ████████████████████              40%  (200)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🗂️ Dataset Schema

| # | Feature | Type | Description | Spam Signal |
|---|---------|------|-------------|-------------|
| 1 | `message_id` | Int | Unique message identifier | — |
| 2 | `message_text` | String | Raw message content | — |
| 3 | `message_length` | Int | Character count of message | — |
| 4 | `word_count` | Int | Total word count | — |
| 5 | `num_urls` | Int | Number of URLs in message | 📈 Higher = More Spam |
| 6 | `num_digits` | Int | Number of digits in message | 📈 Higher = More Spam |
| 7 | `num_special_chars` | Int | Count of special characters | 📈 Higher = More Spam |
| 8 | `spam_keyword_score` | Float | Spam keyword frequency score | ⭐ Top Signal |
| 9 | `legit_keyword_score` | Float | Legitimate keyword score | 📉 Lower = More Spam |
| 10 | `sender_activity_score` | Float | Sender's account activity | 📉 Lower = More Spam |
| 11 | `sender_account_age_days` | Int | Age of sender's account (days) | ⭐ Top Signal |
| 12 | `messages_sent_last_24h` | Int | Messages sent in last 24 hours | 📈 Higher = More Spam |
| 13 | `timestamp` | DateTime | Message send timestamp | → hour, day extracted |
| 14 | `hour_of_day` | Int | Hour when message was sent | — |
| 15 | `day_of_week` | Int | Day of week (0=Mon) | — |
| 16 | `spam_label` | Binary | **0 = Legit, 1 = Spam** | 🎯 **TARGET** |

---

## 🧠 Theory & Concept Cards

> **Each concept below is explained with a visual card — just like the Data Workflow pipeline above.**

---

### 🃏 CARD 01 / PROBABILITY — Conditional Probability

```
╔══════════════════════════════════════════════════════════════════╗
║  CONCEPT CARD 01 / PROBABILITY                                   ║
║  Conditional Probability                                         ║
║  FOUNDATION OF BAYESIAN CLASSIFICATION                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  DEFINITION:                                                     ║
║  P(A|B) = P(A ∩ B) / P(B)                                       ║
║                                                                  ║
║  READ AS: "Probability of A given that B has occurred"           ║
║                                                                  ║
║  SPAM EXAMPLE:                                                   ║
║  ┌─────────────────────────────────────────────────────┐        ║
║  │  P(Spam | "FREE" in message)                        │        ║
║  │  = P(message is Spam AND contains "FREE")           │        ║
║  │  ÷  P(message contains "FREE")                      │        ║
║  │                                                     │        ║
║  │  If 90% of messages with "FREE" are spam →          │        ║
║  │  P(Spam | "FREE") = 0.90                            │        ║
║  └─────────────────────────────────────────────────────┘        ║
║                                                                  ║
║  WHY IT MATTERS:                                                 ║
║  ✓ Updates belief based on observed evidence                     ║
║  ✓ Foundation for Bayes' Theorem                                 ║
║  ✓ Allows probabilistic decision-making                          ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### 🃏 CARD 02 / THEOREM — Bayes' Theorem

```
╔══════════════════════════════════════════════════════════════════╗
║  CONCEPT CARD 02 / THEOREM                                       ║
║  Bayes' Theorem                                                  ║
║  THE ENGINE OF PROBABILISTIC CLASSIFICATION                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  FORMULA:                                                        ║
║  P(Spam|Features) = P(Features|Spam) × P(Spam) / P(Features)   ║
║                     ─────────────────  ──────────  ────────────  ║
║                       Likelihood        Prior      Evidence      ║
║                                                                  ║
║  STEP-BY-STEP FOR A MESSAGE:                                     ║
║  ┌─────────────────────────────────────────────────────┐        ║
║  │  01. Prior:      P(Spam) = 0.40 (40% spam)         │        ║
║  │  02. Likelihood: P(URLs>0 | Spam) = 0.75           │        ║
║  │                  P(URLs>0 | Legit) = 0.10           │        ║
║  │  03. Evidence:   P(URLs>0) = 0.75×0.4 + 0.10×0.6  │        ║
║  │                            = 0.36                   │        ║
║  │  04. Posterior:  P(Spam|URLs>0) = 0.75×0.4 / 0.36 │        ║
║  │                                 = 0.833 → SPAM ✓   │        ║
║  └─────────────────────────────────────────────────────┘        ║
║                                                                  ║
║  CHECKLIST:                                                      ║
║  ✓ Prior computed from training data                             ║
║  ✓ Likelihood from class-conditional PDFs                        ║
║  ✓ Posterior → argmax decides class                              ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### 🃏 CARD 03 / CLASSIFIER — Naive Bayes

```
╔══════════════════════════════════════════════════════════════════╗
║  CONCEPT CARD 03 / CLASSIFIER                                    ║
║  Naive Bayes                                                     ║
║  PROBABILISTIC · FAST · INTERPRETABLE                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  KEY ASSUMPTIONS:                                                ║
║  ┌─────────────────────────────────────────────────────┐        ║
║  │  1. Feature Independence (the "Naive" assumption)   │        ║
║  │     P(x₁,x₂,...,xₙ|class) = ∏ P(xᵢ|class)        │        ║
║  │                                                     │        ║
║  │  2. Gaussian Distribution (GaussianNB)              │        ║
║  │     P(xᵢ|class) = N(μ_class, σ²_class)             │        ║
║  │     Each feature ~ Normal distribution per class    │        ║
║  └─────────────────────────────────────────────────────┘        ║
║                                                                  ║
║  HOW PREDICTION WORKS:                                           ║
║  01 / Compute P(Spam)  and P(Legit)  from training              ║
║  02 / Compute P(xᵢ|Spam) and P(xᵢ|Legit) per feature           ║
║  03 / Multiply likelihoods × prior (log-sum for stability)      ║
║  04 / Predict class with highest posterior                       ║
║                                                                  ║
║  QUALITY METRICS:                                                ║
║  ✓ Accuracy  = 1.0000   ✓ Precision = 1.0000                    ║
║  ✓ Recall    = 1.0000   ✓ AUC-ROC   = 1.0000                    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### 🃏 CARD 04 / CLASSIFIER — K-Nearest Neighbors

```
╔══════════════════════════════════════════════════════════════════╗
║  CONCEPT CARD 04 / CLASSIFIER                                    ║
║  K-Nearest Neighbors (KNN)                                       ║
║  DISTANCE-BASED · LAZY LEARNER · NON-PARAMETRIC                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  WORKING PRINCIPLE:                                              ║
║  ┌─────────────────────────────────────────────────────┐        ║
║  │  01 / STORE   All training points (no model built)  │        ║
║  │  02 / QUERY   New message arrives                   │        ║
║  │  03 / MEASURE Compute distance to all training pts  │        ║
║  │  04 / SELECT  Find K nearest neighbors              │        ║
║  │  05 / VOTE    Majority class wins → prediction      │        ║
║  └─────────────────────────────────────────────────────┘        ║
║                                                                  ║
║  DISTANCE METRICS COMPARED:                                      ║
║  Euclidean:  d = √Σ(xᵢ-yᵢ)²     → Best for continuous features ║
║  Manhattan:  d = Σ|xᵢ-yᵢ|       → Robust to outliers           ║
║  Minkowski:  d = (Σ|xᵢ-yᵢ|^p)^(1/p) → Generalization          ║
║                                                                  ║
║  K ANALYSIS (k=1 to k=21):                                      ║
║  k=1  → Overfit (Train=1.00, Test↓)                             ║
║  k=5  → Best balance ← CHOSEN                                   ║
║  k=21 → Underfit (smoothed boundary)                            ║
║                                                                  ║
║  ✓ Best K = 5  ✓ F1 = 1.000  ✓ AUC = 1.000                     ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### 🃏 CARD 05 / CLASSIFIER — Support Vector Machine

```
╔══════════════════════════════════════════════════════════════════╗
║  CONCEPT CARD 05 / CLASSIFIER                                    ║
║  Support Vector Machine (SVM)                                    ║
║  MARGIN-BASED · KERNEL TRICK · MAXIMUM SEPARATION               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  CORE IDEA — Maximum Margin Hyperplane:                          ║
║  ┌─────────────────────────────────────────────────────┐        ║
║  │   Spam  ●  ●                   ○  ○  Legit          │        ║
║  │       ●   ●  |‾‾‾‾‾‾‾‾‾|  ○   ○                   │        ║
║  │         ●  ←─┤ MARGIN  ├─→ ○                       │        ║
║  │              |_________|                             │        ║
║  │         Support Vectors ↑ (on margin boundary)      │        ║
║  │  Goal: Maximize margin = 2/||w||                    │        ║
║  └─────────────────────────────────────────────────────┘        ║
║                                                                  ║
║  KERNELS EXPLAINED:                                              ║
║  Linear:     K(x,y) = x·y        → Straight boundary            ║
║  RBF:        K(x,y) = exp(-γ||x-y||²) → Circular/radial        ║
║  Polynomial: K(x,y) = (x·y+c)^d  → Curved degree-d boundary    ║
║                                                                  ║
║  KERNEL COMPARISON:                                              ║
║  Linear     → F1=1.000 AUC=1.000 ← Great fit                   ║
║  RBF        → F1=1.000 AUC=1.000 ← Great fit                   ║
║  Polynomial → F1=0.974 AUC=1.000 ← Slightly conservative       ║
║                                                                  ║
║  PREPARATION CHECKLIST:                                          ║
║  ✓ Feature scaling applied (StandardScaler)                      ║
║  ✓ Support vectors identified per class                          ║
║  ✓ Probability=True for AUC computation                         ║
╚══════════════════════════════════════════════════════════════════╝
```

---

### 🃏 CARD 06 / COMPARISON — Classifier Taxonomy

```
╔══════════════════════════════════════════════════════════════════╗
║  CONCEPT CARD 06 / COMPARISON                                    ║
║  Classifier Taxonomy                                             ║
║  DISTANCE vs MARGIN vs PROBABILISTIC                             ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  DISTANCE-BASED          MARGIN-BASED         PROBABILISTIC      ║
║  ┌─────────────┐         ┌─────────────┐      ┌─────────────┐   ║
║  │    KNN      │         │    SVM      │      │ Naive Bayes │   ║
║  │             │         │             │      │             │   ║
║  │ Similarity  │         │ Hyperplane  │      │  Bayes'     │   ║
║  │ in feature  │         │ + Max       │      │  Theorem    │   ║
║  │ space       │         │   Margin    │      │  P(c|x)     │   ║
║  └─────────────┘         └─────────────┘      └─────────────┘   ║
║                                                                  ║
║  PROPERTY         KNN          SVM          NAIVE BAYES          ║
║  ─────────────────────────────────────────────────────────────  ║
║  Core idea        Distance     Max margin   Bayes Theorem        ║
║  Training         None (lazy)  Optimization MLE parameters       ║
║  Scaling needed   ✅ Yes       ✅ Yes       ❌ No                ║
║  Non-linearity    Inherent     Via kernels  Limited              ║
║  Interpretable    Low          Medium       ✅ High (probs)      ║
║  Speed predict    Slow O(nd)   Fast O(1)    Fast O(d)            ║
║  Imbalance sens.  High         Medium       Low                  ║
║                                                                  ║
║  PRODUCTION CHOICE:                                              ║
║  ✓ SVM RBF for real-time classification (fast inference)         ║
║  ✓ Naive Bayes for probability-based risk scoring                ║
║  ✓ KNN for interpretable local neighborhood analysis             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## ⚙️ ML Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║              MESSAGE INTELLIGENCE SYSTEM — ML PIPELINE                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  01 / INPUT           02 / PROCESS         03 / SPLIT                   ║
║  ┌───────────┐        ┌───────────┐        ┌───────────────────┐        ║
║  │ messages  │   →    │  Encode   │   →    │  Stratified       │        ║
║  │  .csv     │        │  Scale    │        │  Train: 400 (80%) │        ║
║  │ 500 rows  │        │  Feature  │        │  Test:  100 (20%) │        ║
║  └───────────┘        │  Engineer │        └───────────────────┘        ║
║                       └───────────┘                │                    ║
║                                                     │                   ║
║  ┌─────────────────────────────────────────────────▼──────────┐        ║
║  │                  StandardScaler (fit on train only)         │        ║
║  └────┬──────────────────────────┬──────────────────┬─────────┘        ║
║       │                          │                  │                   ║
║  ┌────▼──────┐          ┌────────▼────┐     ┌───────▼──────┐           ║
║  │   KNN     │          │    SVM      │     │  Naive Bayes │           ║
║  │  k=5      │          │ Lin/RBF/Poly│     │  GaussianNB  │           ║
║  │ Euclidean │          │ C=1.0       │     │  P(x|class)  │           ║
║  └────┬──────┘          └────────┬────┘     └───────┬──────┘           ║
║       │                          │                  │                   ║
║  ┌────▼──────────────────────────▼──────────────────▼──────────┐       ║
║  │  Evaluation: Confusion Matrix · AUC-ROC · F1 · P-R Curve    │       ║
║  └──────────────────────────────────────────────────────────────┘       ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 📏 KNN — Distance-Based Classifier

### K Value Analysis

```
K vs PERFORMANCE (trained on 400 samples, tested on 100):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
K=1   Train=1.00  Test=1.00  F1=1.000  → Perfect (but risky)
K=3   Train=0.99  Test=1.00  F1=1.000  → Excellent
K=5   Train=0.99  Test=1.00  F1=1.000  → ✅ CHOSEN (stable)
K=7   Train=0.98  Test=0.99  F1=0.987  → Good
K=11  Train=0.97  Test=0.98  F1=0.973  → Slight smoothing
K=21  Train=0.95  Test=0.97  F1=0.961  → Underfit tendency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Distance Metric Comparison (k=5)

| Distance Metric | Formula | Accuracy | F1-Score | AUC |
|-----------------|---------|:--------:|:--------:|:---:|
| **Euclidean** ← chosen | √Σ(xᵢ-yᵢ)² | 1.0000 | 1.0000 | 1.0000 |
| Manhattan | Σ\|xᵢ-yᵢ\| | 1.0000 | 1.0000 | 1.0000 |
| Minkowski | (Σ\|xᵢ-yᵢ\|^p)^(1/p) | 1.0000 | 1.0000 | 1.0000 |

---

## ⚡ SVM — Margin-Based Classifier

### Kernel Analysis

| Kernel | Decision Boundary | C | Accuracy | Precision | Recall | F1 | AUC |
|--------|------------------|:---:|:--------:|:---------:|:------:|:--:|:---:|
| **Linear** | Straight hyperplane | 1.0 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| **RBF** ← best | Radial/circular | 1.0 | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| **Polynomial** | Degree-3 curve | 1.0 | 0.9800 | 1.0000 | 0.9500 | 0.9744 | 1.0000 |

### Support Vector Analysis (Linear Kernel)

```
Support vectors define the decision boundary:
  Legitimate class: N support vectors (on +margin boundary)
  Spam class:       N support vectors (on -margin boundary)
  
  Margin = 2 / ||w||  ← maximized during training
  Points NOT on margin → ignored for prediction (sparse solution)
```

---

## 📊 Naive Bayes — Probabilistic Classifier

### Manual Bayes Theorem Computation

```
TRAINING PRIORS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P(Spam)       = 0.4000  (200 spam / 500 total)
P(Legitimate) = 0.6000  (300 legit / 500 total)

CLASS-CONDITIONAL STATISTICS (spam_keyword_score, standardized):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P(x | Spam):  μ = +1.112, σ = 0.714  ← High spam_keyword_score
P(x | Legit): μ = −0.741, σ = 0.621  ← Low  spam_keyword_score

MANUAL PREDICTION — Sample Message (feat = +1.321, actual = Spam):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P(x|Spam)  = N(+1.321; μ=+1.112, σ=0.714) = 0.538
P(x|Legit) = N(+1.321; μ=−0.741, σ=0.621) = 0.000

Posterior P(Spam|x)  = 0.538 × 0.40 / norm = 1.000  → SPAM ✅
Posterior P(Legit|x) = 0.000 × 0.60 / norm = 0.000
Model output: P(Spam) = 1.0000 ← matches manual calculation!
```

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

> **Note:** The dataset features (especially `spam_keyword_score` and `sender_account_age_days`) are highly discriminative and linearly separable — all three classifier paradigms achieve perfect separation. This validates the dataset design and feature quality.

### AUC-ROC Visual

```
ALL MODELS — AUC-ROC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNN (k=5)      ██████████████████████████████████████████ 1.0000
SVM Linear     ██████████████████████████████████████████ 1.0000
SVM RBF        ██████████████████████████████████████████ 1.0000
Naive Bayes    ██████████████████████████████████████████ 1.0000
SVM Poly       █████████████████████████████████████████  1.0000 (AUC perfect, F1 slight miss)
```

### Best-for-Use-Case Guide

| Use Case | Best Model | Reason |
|----------|-----------|--------|
| **High Precision** (no false spam flags) | SVM Linear | Perfect precision, interpretable margin |
| **High Recall** (catch all spam) | Any model | All achieve 100% recall |
| **Probability scores** (risk ranking) | Naive Bayes | Posterior P(Spam\|message) directly interpretable |
| **Real-time production** | SVM RBF | Fast inference O(sv × d) after training |
| **Explainability** | Naive Bayes | P(class\|features) with Bayes theorem traceable |

---

## 📈 Visualizations — 10 Charts

| # | Chart | Description | Key Insight |
|:---:|-------|-------------|-------------|
| 01 | `01_eda_distributions.png` | Class pie + 7 feature histograms | Clear bimodal separation for spam_keyword_score |
| 02 | `02_correlation_heatmap.png` | Full 13-feature correlation matrix | spam_kw_score and legit_kw_score strongly anti-correlated |
| 03 | `03_feature_by_class.png` | Boxplots: each feature by spam label | Near-zero overlap confirms linear separability |
| 04 | `04_knn_analysis.png` | K-value curves + distance metrics + CM | k=5 optimal, all distance metrics equivalent |
| 05 | `05_svm_analysis.png` | 3 kernels metrics + CM + KNN vs SVM | Linear and RBF both achieve perfect separation |
| 06 | `06_naive_bayes.png` | Gaussian PDFs + posterior probs + CM | Class-conditional distributions clearly separated |
| 07 | `07_roc_comparison.png` | ROC curves all models + metrics bar | All models hug the top-left corner |
| 08 | `08_precision_recall.png` | P-R curves + F1/Rec/AUC final bars | All models maintain high precision at all recall levels |
| 09 | `09_bayes_probability.png` | Bayes PDFs + P(Spam\|x) + CM | Manual Bayes matches model perfectly |
| 10 | `10_final_diagnostics.png` | Final CMs + Classifier taxonomy card | Visual summary of all 3 paradigms |

---

## 📁 Project Structure

```
📨 message-intelligence-system/
│
├── 📓 Message_Intelligence_System.ipynb     ← Pre-executed notebook (all outputs)
│
├── 📊 data/
│   ├── messages.csv                          ← Dataset (500 × 16 columns)
│   └── final_predictions.csv                ← ✅ Predictions from all 4 models
│                                                (100 test rows × 21 columns)
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
└── 📄 README.md                              ← This file
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Dhairyapatel1mc/message-intelligence-system.git
cd message-intelligence-system
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

### 3. Run Notebook

```bash
jupyter notebook Message_Intelligence_System.ipynb
```

### 4. Quick Inference

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Choose your classifier
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_sc, y_train)
svm = SVC(kernel='rbf', probability=True).fit(X_train_sc, y_train)
nb  = GaussianNB().fit(X_train_sc, y_train)

# Predict
for msg_id, prob in zip(message_ids, svm.predict_proba(X_new)[:,1]):
    label = "🚨 SPAM" if prob >= 0.5 else "✅ LEGIT"
    print(f"Message {msg_id}: {label} | P(spam)={prob:.4f}")
```

---

## 📋 Final Report

### Strengths & Weaknesses

```
╔══════════════════════════════════════════════════════════════════════════╗
║                   CLASSIFIER STRENGTHS & WEAKNESSES                      ║
╠═══════════════╦════════════════════════════════════════════════════════╣
║     KNN       ║  ✅ No training, intuitive, adapts to local geometry   ║
║               ║  ❌ Slow at inference O(n×d), sensitive to scaling     ║
╠═══════════════╬════════════════════════════════════════════════════════╣
║  SVM Linear   ║  ✅ Fast inference, interpretable weights              ║
║               ║  ❌ Poor on non-linear data without kernel trick        ║
╠═══════════════╬════════════════════════════════════════════════════════╣
║   SVM RBF     ║  ✅ Handles non-linearity, robust to noise             ║
║               ║  ❌ Slow training O(n²-n³), kernel parameter tuning    ║
╠═══════════════╬════════════════════════════════════════════════════════╣
║  Naive Bayes  ║  ✅ Probabilistic output, fast, handles imbalance      ║
║               ║  ❌ Independence assumption rarely holds in practice    ║
╚═══════════════╩════════════════════════════════════════════════════════╝
```

### Top Discriminating Features

```
spam_keyword_score       ████████████████████████████  Most important
legit_keyword_score      ████████████████████████████  Anti-correlated
sender_account_age_days  ██████████████████████        Spam = new accts
messages_sent_last_24h   █████████████████             Spam = high freq
num_urls                 ████████████                  Spam = more URLs
sender_activity_score    ██████████                    Spam = low activity
```

### Business Recommendation

- 🚀 **Deploy SVM RBF** as the primary real-time classifier
- 📊 **Use Naive Bayes** for spam probability scoring and risk tiers
- 🔍 **Monitor** `spam_keyword_score` and `sender_account_age_days` as top features
- 🔄 **Retrain weekly** as spam patterns evolve
- ⚡ **Set threshold at 0.4** to catch borderline cases (lower FN risk)

---

## 👤 Author & Contact

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=6,14,18&height=120&section=header&text=Ghost%20(Patel%20Dhairya)&fontSize=30&fontColor=ffffff&desc=Data%20Scientist%20%7C%20ML%20Engineer%20%7C%20RWSkill&descSize=14&animation=fadeIn" width="70%"/>

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

## 🛠️ Tech Stack

<div align="center">

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **ML Models** | Scikit-Learn (KNN, SVC, GaussianNB) |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, SciPy |
| **Notebook** | Jupyter Notebook (pre-executed) |
| **Preprocessing** | StandardScaler, StratifiedKFold |

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
⭐ If this project helped you, please star the repository!
```

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,18,24&height=100&section=footer&animation=fadeIn" width="100%"/>

*Built with precision by Ghost (Patel Dhairya) · Red and White Skill Education (RWSkill) · 2024*

</div>
