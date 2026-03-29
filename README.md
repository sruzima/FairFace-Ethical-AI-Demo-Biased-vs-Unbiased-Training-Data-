# FairFace Ethical AI Demo: Biased vs Unbiased Training Data

This repo demonstrates how **biased training data** can produce **unequal model performance across demographic groups** and how training on **unbiased/original data** reduces that harm. Using the **FairFace** dataset, we train two **ResNet50** gender-classification models and measure fairness with **race** and **race × gender** metrics.


## What’s in this repo

### Models
- **Model A (Biased):** trained on an intentionally skewed subset (`biased_train.csv`)  
  - Output: `modelA_resnet50_biased.pth`
- **Model U (Unbiased):** trained on original/unbiased training data (`train_unbiased.csv`)  
  - Output: `modelU_resnet50_unbiased.pth`

### Data splits / labels (CSV)
- `biased_train.csv` — biased training split used for Model A  
- `train_unbiased.csv` — unbiased training split used for Model U  
- `val.csv` — validation split  
- `fairface_label_train.csv`, `fairface_label_val.csv` — original FairFace label files (if included)

### Notebook
End-to-end Kaggle notebook that:
- creates biased/unbiased splits  
- trains both models  
- evaluates fairness  
- exports `.pth` model files + CSV splits  


## Why this matters (Ethical AI & Governance)

Overall accuracy can look “good” while some demographic groups experience **higher error rates**. This repo shows:

- how representation bias (who appears in training data) affects model outcomes  
- why you must evaluate **by group**, not only overall  
- how to report **worst-group performance** and **disparity gaps**  
- intersectional effects (e.g., **race × gender**) similar to **Gender Shades** style analysis  

---

## Fairness metrics reported

We compute group-disaggregated metrics and bias summaries:

### By race
- Accuracy by race  
- Worst-group accuracy  
- Disparity gap (max − min)  

### Intersectional (race × gender)
- Accuracy by race × gender  
- Worst intersectional group accuracy  
- Intersectional disparity gap  

### Optional (governance-style)
- FPR / FNR by group  
- Equalized odds gaps (TPR/FPR gaps)  
- NIST-style reporting: TPR at fixed FPR using ROC curves (when probabilities are available)  

---

## How to run (recommended: Kaggle)

1. Open the notebook in Kaggle.  
2. Add the dataset:
   - `fairface-img-margin025` (recommended) or `margin125`
   - `fairface_label_train.csv`, `fairface_label_val.csv`
3. Set paths (example):
   - `DATASET_DIR=/kaggle/input/fairface_aml`
   - `IMAGE_ROOT=/kaggle/input/fairface_aml/fairface-img-margin025`
4. Run all cells.  
5. Download outputs from the Kaggle **Output** panel:
   - `modelA_resnet50_biased.pth`
   - `modelU_resnet50_unbiased.pth`
   - `biased_train.csv`, `train_unbiased.csv`, `val.csv` (if exported)

---

## Deployment (Replit / Gradio)

You can deploy a lightweight demo app using:

- `modelA_resnet50_biased.pth`
- `modelU_resnet50_unbiased.pth`
- (optional) `biased_train.csv` / `train_unbiased.csv` to show training-data bias charts

Recommended UI:
- image upload  
- side-by-side prediction comparison (Biased vs Unbiased)  
- dataset bias explorer (race × gender charts)  

If you want, I can provide a ready-to-run `app.py` + `requirements.txt` for Replit.

---

## Notes & limitations
- This project predicts **gender** as an attribute classification task and is intended for **educational bias measurement**, not real-world identity verification.  
- Demographic labels may contain noise and may not represent all identities.  
- Face-based models can be harmful in high-stakes uses; this repo is focused on **measurement + governance learning**.  

---

## Credits
FairFace dataset and labels: created by the FairFace authors (see the original FairFace repository and paper for details).
