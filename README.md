# Pneumonia-ChestXray-Classification

### Model Evaluation Results

The CNN model was evaluated on a held-out test dataset containing 624 chest X-ray images: 234 NORMAL and 390 PNEUMONIA.

| Metric | NORMAL | PNEUMONIA |
|---|---:|---:|
| Precision | 0.97 | 0.73 |
| Recall | 0.38 | 0.99 |
| F1-score | 0.55 | 0.84 |

**Overall test accuracy:** 76%  
**Weighted F1-score:** 0.73

The model demonstrates high recall for the PNEUMONIA class but has limited recall for the NORMAL class. This indicates that many normal images are incorrectly classified as pneumonia. Further investigation of preprocessing, class predictions, decision threshold selection, and model training is required.

These results are experimental and do not establish clinical diagnostic performance.
