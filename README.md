# Enhancing AI Acceptance and Policy Impact Using SHAP and Explainable AI (XAI)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange?logo=python&logoColor=white)](https://shap.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end Explainable AI (XAI) framework** that applies SHAP to breast cancer diagnosis,
> measures the impact of explanations on clinical trust, and generates a policy recommendation
> report for AI regulation in healthcare.

---

## Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [Dataset](#-dataset)
- [Models](#-models)
- [SHAP Visualizations](#-shap-visualizations)
- [Trust Study](#-trust-study)
- [Policy Report](#-policy-report)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deliverables](#-deliverables)
- [References](#-references)
- [License](#-license)

---

## Overview

Despite strong diagnostic performance, AI models in healthcare face a critical adoption
barrier: **lack of transparency**. Clinicians, patients, and regulators cannot trust
decisions they cannot understand.

Black-box AI Model
↓
SHAP Analysis
↓
Feature-level Explanations
↓
Trust Measurement Study
↓
Policy Recommendation Report


**Outcomes:**
-  Breast cancer diagnosis model with **AUC > 0.97**
-  Full SHAP explanation suite (global + local + dependence)
-  Simulated trust study showing **~92% trust improvement** after XAI
-  Auto-generated 7-section **PDF policy report**

---

##  Project Structure

xai-shap-healthcare/
│
├──  XAI_SHAP_Pipeline.ipynb       
│
├──  xai_outputs/                  
│   ├── 01_dataset_overview.png      
│   ├── 02_model_comparison.png     
│   ├── 03_shap_bar_global.png       
│   ├── 04_shap_beeswarm.png         
│   ├── 05_shap_dependence.png       
│   ├── 06_shap_force_plots.png     
│   ├── 07_trust_study.png          
│   └── XAI_Policy_Report.pdf        
│
├──  requirements.txt              
├──  README.md                     
└──  LICENSE                       


---

##  Workflow

The project follows a 5-phase pipeline.

Phase 1 ── Dataset Loading & EDA
└─ Wisconsin Breast Cancer Dataset
└─ Class balance, feature correlation

Phase 2 ── Model Training & Selection
└─ Logistic Regression, Random Forest,
Gradient Boosting, XGBoost
└─ 5-Fold CV + ROC comparison

Phase 3 ── SHAP Explainability
└─ TreeExplainer / LinearExplainer
└─ Summary, Beeswarm, Dependence, Force plots

Phase 4 ── Trust Measurement Study
└─ Before/After SHAP exposure
└─ 4 stakeholder groups (N=120)

Phase 5 ── Policy Report Generation
└─ Auto-generated PDF
└─ 7 policy recommendations


---

##  Dataset

| **Property**     | **Detail**                                      |
|------------------|-------------------------------------------------|
| Name             | Wisconsin Breast Cancer Diagnostic Dataset      |
| Source           | `sklearn.datasets.load_breast_cancer()`         |
| Samples          | 569                                             |
| Features         | 30 numerical (cell nucleus measurements)        |
| Classes          | Malignant (1) / Benign (0)                      |
| Class Balance    | ~37% Malignant / ~63% Benign                    |
| Preprocessing    | StandardScaler normalization                    |
| Split            | 80% Train / 20% Test (stratified)               |

**Top diagnostic features identified by SHAP:**
- `worst radius`
- `worst perimeter`
- `mean concave points`
- `worst concave points`
- `worst area`

---

##  Models

Four classifiers were trained and compared:

| **Model**             | **CV AUC (5-Fold)**  | **Test AUC** | **Test Accuracy** |
|-----------------------|----------------------|--------------|-------------------|
| Logistic Regression   | ~0.990 ± 0.007       | ~0.994       | ~96.5%            |
| Random Forest         | ~0.993 ± 0.005       | ~0.995       | ~97.4%            |
| Gradient Boosting     | ~0.994 ± 0.004       | ~0.996       | ~97.4%            |
| **XGBoost**           | **~0.996 ± 0.003**   | **~0.997**   | **~98.2%**        |

> The best-performing model is automatically selected and used for all SHAP analysis.

---

##  SHAP Visualizations

### 1. Global Feature Importance (Bar Plot)
Shows the **average impact** of each feature across all predictions.
Answers: *"Which features matter most overall?"*

### 2. Beeswarm Plot
Shows **direction and magnitude** of each feature's impact.
Red = high feature value | Blue = low feature value.
Answers: *"How does each feature push predictions?"*

### 3. Dependence Plots
Shows the **relationship** between a feature's value and its SHAP value.
Answers: *"At what threshold does this feature become dangerous?"*

### 4. Force Plots (Individual Cases)
Shows **per-patient explanations** for:
-  True Positive — correctly identified malignant
-  True Negative — correctly identified benign
-  False Positive — incorrectly flagged as malignant

Answers: *"Why did the model make THIS specific decision?"*

---

##  Trust Study

A simulated before/after study measured AI trust across **4 stakeholder groups**:

| **Group**          | **N** | **Trust Before** | **Trust After** | **Change**  |
|--------------------|-------|------------------|-----------------|-------------|
| Clinicians         | 40    | ~4.2 / 10        | ~7.8 / 10       | +3.6 (+86%) |
| Radiologists       | 35    | ~4.8 / 10        | ~8.1 / 10       | +3.3 (+69%) |
| Medical Students   | 25    | ~3.5 / 10        | ~7.2 / 10       | +3.7 (+106%)|
| Policy Makers      | 20    | ~3.0 / 10        | ~6.8 / 10       | +3.8 (+127%)|
| **Overall**        | **120**| **~3.9 / 10**   | **~7.5 / 10**   | **+92%**    |

> Scale: 1 = No trust at all → 10 = Complete trust
> Methodology based on: Tonekaboni et al. (2019), Holzinger et al. (2020)

---

## References
1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
2. Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What clinicians want: contextualizing explainable machine learning for clinical end use. MLHC.
3. Holzinger, A., Langs, G., Denk, H., Zatloukal, K., & Müller, H. (2019). Causability and explainability of AI in medicine. WIREs Data Mining.
4. Arrieta, A. B., et al. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges. Information Fusion.
5. European Commission. (2021). Proposal for a Regulation on Artificial Intelligence (AI Act).
6. FDA. (2021). Artificial Intelligence/Machine Learning-Based Software as a Medical Device Action Plan.

This project addresses that gap by building a full XAI pipeline.
