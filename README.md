## Model Evaluation

The trained CNN was evaluated on 624 test images.

| Metric | NORMAL | PNEUMONIA |
|---|---:|---:|
| Precision | 95.10% | 73.75% |
| Recall | 41.45% | 98.72% |
| F1-score | 57.74% | 84.43% |
| Test support | 234 | 390 |

- **Overall test accuracy:** 77.24%
- **Weighted F1-score:** 74.42%

### Confusion Matrix

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 97 | 137 |
| PNEUMONIA | 5 | 385 |

The model detects most pneumonia examples in this test set, but it incorrectly classifies some NORMAL images as PNEUMONIA. These results are from this dataset and should not be interpreted as evidence of clinical performance.

### Disclaimer

This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose disease or make healthcare decisions.