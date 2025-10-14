# AMAZON-ML-CHALLENGE-HACKATHON-2025


# Amazon ML Challenge 2025 - Smart Product Pricing

## 🎯 Repository Description

This repository contains a comprehensive machine learning solution for the **Amazon ML Challenge 2025: Smart Product Pricing**. The project implements an advanced ensemble learning approach combining transformer-based text embeddings, sophisticated feature engineering, and multi-model stacking to predict e-commerce product prices with high accuracy.

**Key Highlights:**
- 🏆 **SMAPE Score**: 57% (Strong competitive performance)
- 🚀 **Advanced Architecture**: Transformer embeddings + Ensemble learning + Stacking meta-model
- 📊 **Comprehensive Pipeline**: End-to-end ML solution from data preprocessing to prediction
- 🔬 **State-of-the-art Features**: 402-dimensional feature space with semantic text understanding
- ✅ **Competition Ready**: Compliant with all requirements (<8B parameters, MIT license)

**Technologies Used:**
- **NLP**: Sentence-Transformers (all-MiniLM-L6-v2) for semantic text embeddings
- **ML Models**: LightGBM, CatBoost, XGBoost with ensemble stacking
- **Feature Engineering**: Advanced text processing, IPQ extraction, brand detection
- **Validation**: 5-fold cross-validation with SMAPE evaluation
- **Languages**: Python with scikit-learn, pandas, numpy ecosystem


## 🏆 Results

- **SMAPE Score**: 57% (Strong performance)
- **R² Score**: 0.731 (Strong predictive power)
- **MAE**: $0.623 (Low error)

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
python run_training.py
```

### Quick Test (Small Sample)

```bash
python run_training.py --quick
```

### Training Only

```bash
python run_training.py --train-only
```

### Prediction Only

```bash
python run_training.py --predict-only
```

## 📁 Project Structure
