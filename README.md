# TriMixGen-Indic Shared Task: Token-Level Language Identification (LID)

This repository contains the fine-tuning and inference pipeline for token-level Language Identification (LID) on code-mixed text, developed for the **TriMixGen-Indic Shared Task (Subtask B)** hosted on Codabench. 

The system fine-tunes multilingual transformer models (optimized for MuRIL) to tag code-mixed sentences across language pairs (e.g., English-Hindi-Gujarati and English-Hindi-Bengali) with fine-grained token labels: `HIN`, `BEN`, `GUJ`, `ENG`, and `UNI` (universal/punctuation/symbols).

---

## Key Features & Methodological Improvements

1. **MuRIL Backbone (`google/muril-base-cased`):** Pretrained extensively on transliterated and native script Indian text, offering superior performance on romanized code-mixed data compared to standard multilingual models like XLM-R.
2. **Class-Weighted Loss:** Handles extreme class imbalance (e.g., frequent English tokens vs. rare code-switched Indian language tokens) using inverse-frequency weighting to optimize directly for the task's official metric: **Macro F1**.
3. **K-Fold Cross-Validation:** Implements robust out-of-fold evaluation (default $K=5$) to mitigate variance on small datasets (~480 sentences per pair), preventing overfitting and providing trustworthy validation signals.
4. **Model Ensembling:** Supports multi-model probability averaging across all cross-validation folds at inference time for a substantial boost in generalization and Macro F1.
5. **Regularization:** Incorporates learning rate warmup, cosine decay schedules, label smoothing, and gradient capping to stabilize training on low-resource fine-tuning sets.

---

## Project Structure

```text
├── xlmr_lid.py               # Main training and prediction script
├── README.md                 # Project documentation
└── submissions/              # Generated final submission zip and conll files

