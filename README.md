# MNIST Digit Recognizer

Draw a digit (0–9) with your mouse and a CNN ensemble predicts it.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501.

## Model
- Ensemble of 10 CNNs (5 multi-branch, 5 VGG-style), averaged softmax outputs.
- Trained on a seed-42 split of the pooled 70k MNIST images: 56,700 train / 6,300 validation / 7,000 test.
- Augmentation: rotation ±10°, zoom ±10%, shift ±10%. Label smoothing 0.05, Adam with cosine decay, 45 epochs.

| | Errors | Accuracy |
|---|---|---|
| Validation (6,300) | 17 | 99.73% |
| Test (7,000) | 16 | 99.77% |

No checkpoint or ensemble choice was made using the test set.

## Files
- `app.py` — Streamlit app. Drawn digits are cropped, scaled to 20 px, centred by centre of mass in 28×28 (the MNIST recipe), then normalised.
- `mnist_ensemble.keras` — the ensemble model, input (28, 28, 1), output 10 probabilities.
- `preprocess.json` — mean/std used for normalisation.
