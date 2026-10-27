# MNIST Digit Recognizer

Draw a digit (0–9) with your mouse and a CNN ensemble predicts it.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501.

## Model
- Ensemble of 20 CNNs (8 multi-branch, 8 VGG-style, 4 pre-activation ResNet), averaged softmax outputs.
- Weights stored in float16 to fit GitHub's file limit (predictions identical to float32 on all 13,300 val/test images).
- Trained on a seed-42 split of the pooled 70k MNIST images: 56,700 train / 6,300 validation / 7,000 test.
- Augmentation: rotation ±10°, zoom ±10%, shift ±10%; elastic distortion on half the images for the 10 round-3 members. Label smoothing 0.05, Adam with warm-up + cosine decay, 45–50 epochs.

| | Errors | Accuracy |
|---|---|---|
| Validation (6,300) | 13 | 99.79% |
| Test (7,000) | 18 | 99.74% |

The ensemble was chosen on validation error by a rule fixed in advance; the test set was read only to report results (2 reads in total).

## Files
- `app.py` — Streamlit app. Drawn digits are cropped, scaled to 20 px, centred by centre of mass in 28×28 (the MNIST recipe), then normalised.
- `mnist_ensemble.keras` — the ensemble model, input (28, 28, 1), output 10 probabilities.
- `preprocess.json` — mean/std used for normalisation.
