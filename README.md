# **Toxic Comment Classification with Fairness-Aware NLP**

This project implements an end-to-end toxic speech classification pipeline that evaluates **both accuracy and equity**. We compare transformer-based models (RoBERTa, BERT), a classical tree-based model (XGBoost), and a sequential model (LSTM), while auditing **identity subgroup fairness** using Jigsaw’s unintended bias framework.

---

## **Project Objective**

Toxic language online disproportionately harms vulnerable communities. While LLMs like BERT and RoBERTa have improved text classification, they risk reinforcing societal bias. This project aims to:

- Detect toxic comments in online conversations  
- Evaluate and mitigate **bias against identity subgroups** (race, gender, religion, sexuality, mental health)  
- Compare **model families**: Transformers, XGBoost, LSTM  
- Use **fairness-aware metrics**: subgroup AUC, BPSN AUC, BNSP AUC  

---

## **Dataset**

- **Source**: Jigsaw Unintended Bias in Toxicity Classification (Kaggle)  
- **Volume**: ~1.8M Wikipedia comments  
- **Targets**: Binary toxicity label (`0` = non-toxic, `1` = toxic)  
- **Identity Attributes**: Annotations across 8+ dimensions (e.g., *female*, *black*, *Christian*, *mental illness*)  
- **Challenge**: Multi-label identity data, label imbalance, subtle toxicity, societal bias  

---

## **Modeling Approaches**

### 1. **RoBERTa Fine-Tuning**
- Architecture: `RobertaForSequenceClassification` via Hugging Face  
- Tokenizer: `RobertaTokenizerFast`  
- Training: Sigmoid output, `BCEWithLogitsLoss`, warm-up scheduler  
- Strength: Deeper pretraining, strong contextual understanding  

### 2. **BERT Fine-Tuning**
- Architecture: `BertForSequenceClassification`  
- Model: `'bert-base-uncased'`  
- Training setup mirrors RoBERTa  
- Motivation: Compare with RoBERTa for generalizability and bias sensitivity  

### 3. **XGBoost Baseline**
- Features: TF-IDF, Word2Vec averages  
- Fast, interpretable model to benchmark LLM gains  
- Weakness: Poor recall on toxic class due to feature sparsity  

### 4. **LSTM (Word2Vec Embedding Layer)**
- Architecture: Bi-LSTM → Dropout → Sigmoid  
- Embeddings: Pre-trained Gensim Word2Vec  
- Strength: Captures sequence dependencies for implicit toxic tone  

---

## **Tech Stack**

### NLP & Modeling
- `transformers` (Hugging Face): BERT & RoBERTa  
- `torch`, `torch.nn`, `DataLoader`, `CUDA`: Model training + GPU acceleration  
- `xgboost`: Gradient boosting tree model  
- `gensim`: Word2Vec support for LSTM/XGBoost  
- `nltk`, `emoji`, `re`: Tokenization + emoji/punctuation removal  

### Evaluation & Fairness
- `scikit-learn`: F1, precision, recall, confusion matrix, AUC  
- `matplotlib`, `seaborn`: ROC and fairness visualization  
- Jigsaw fairness metrics:
  - `subgroup_auc`
  - `bpsn_auc`
  - `bnsp_auc`  

---
## Performance Summary

| Model      | Accuracy | Toxic F1 | AUC     | Fairness Score | Strengths                                             |
|------------|----------|----------|---------|----------------|-------------------------------------------------------|
| **XGBoost**| **94.4%**| 0.16     | **0.85**| —              | Highest accuracy & AUC; fast & interpretable baseline |
| BERT       | 79%      | 0.64     | 0.75    | 0.684          | Balanced recall + stronger fairness                   |
| RoBERTa    | 78%      | 0.64     | 0.75    | **0.697**      | Best fairness-aware metric                            |
| LSTM       | 75%      | 0.58     | 0.72    | 0.660*         | Lightweight; captures temporal flow                   |

---

Fairness metrics show that despite strong overall accuracy, **models like XGBoost underperform on toxic recall**, potentially overlooking harmful content. In contrast, **transformer models** offer more balanced treatment across identity groups but come with higher computational costs.

---

## Key Takeaways

- **Best Raw Performance**: **XGBoost** achieves the highest **accuracy (94.4%)** and **AUC (0.85)**, making it a strong benchmark — but **poor toxic recall (F1 = 0.16)** limits its use in sensitive applications.
- **Best Fairness Tradeoff**: **RoBERTa** has the highest **fairness-aware score (0.697)** and performs well across multiple identity groups.
- **Best Toxic Sensitivity**: **BERT** offers the most **balanced performance** with strong toxic recall (0.65) and consistent subgroup metrics.
- **Most Efficient Model**: **LSTM** is compact and reasonably fair, though its performance lags behind larger models.

> **Final Recommendation**: Use **RoBERTa** or **BERT** for production deployments where fairness and safety are critical. Use **XGBoost** as a fast, interpretable baseline — not for moderation-sensitive tasks.


