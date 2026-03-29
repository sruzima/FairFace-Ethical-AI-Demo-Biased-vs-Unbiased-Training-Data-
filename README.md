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
- `fairface_label_train.csv`, `fairface_label_val.csv` — original FairFace label files

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
 
  
## How to run (recommended workflow)

1. Open the project notebook in your preferred environment (e.g., Jupyter Notebook, VS Code, Kaggle, or Google Colab).

2. Make sure the FairFace data is available locally or mounted in your environment, including:
   - Image folders (e.g., `fairface-img-margin025` or `fairface-img-margin125`)
   - Label files: `fairface_label_train.csv`, `fairface_label_val.csv`

3. Update the dataset paths in the notebook to match your environment. For example:
   - `DATASET_DIR=<path-to-your-fairface-dataset-root>`
   - `IMAGE_ROOT=<path-to-your-selected-image-folder>`

4. Run the notebook cells from top to bottom (or use “Run All”).

5. After training finishes, collect the generated artifacts (exact filenames may vary depending on your configuration):
   - Model checkpoints: `modelA_resnet50_biased.pth`, `modelU_resnet50_unbiased.pth`
   - Exported split CSVs (if enabled): `biased_train.csv`, `train_unbiased.csv`, `val.csv`

6. (Optional) Use the saved `.pth` files to deploy a small inference demo (e.g., Gradio on Replit) for live bias vs unbiased comparison.


## Deployment (Replit / Gradio or any prefered options)

You can deploy a lightweight demo app using:

- `modelA_resnet50_biased.pth`
- `modelU_resnet50_unbiased.pth`
- (optional) `biased_train.csv` / `train_unbiased.csv` to show training-data bias charts

Recommended UI:
- image upload  
- side-by-side prediction comparison (Biased vs Unbiased)  
- dataset bias explorer (race × gender charts)  


## Notes & limitations
- This project predicts **race** as an attribute classification task and is intended for **educational bias measurement and governance learning**, not real-world identity verification.  
- Fairness analysis is reported for both **race** groups and **intersectional groups (race × gender)** to highlight how disparities can appear at multiple levels.  
- Demographic labels may contain noise, may be imperfect proxies, and may not represent all identities or contexts.  
- Face-based models can be harmful in high-stakes uses; this repo focuses on **measurement, transparency, and responsible evaluation practices** rather than deployment for real-world decision-making.


## Credits
FairFace dataset and labels: created by the FairFace authors (see the original FairFace repository and paper for details at https://github.com/joojs/fairface ).
