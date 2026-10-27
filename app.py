"""Streamlit interface for the MNIST handwritten digit recognizer."""

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


MODEL_PATH = Path(__file__).with_name("mnist_ensemble.keras")
PREPROCESS_PATH = Path(__file__).with_name("preprocess.json")


def preprocess_canvas(rgba, mean, std):
    """Turn a drawn canvas (HxWx4 uint8, white on black) into MNIST-style model input.

    Returns (model_input (1,28,28,1) float32, preview 28x28 uint8), or (None, None)
    when the canvas is effectively empty.
    """
    rgba = np.asarray(rgba)
    gray = rgba[..., :3].max(axis=-1).astype(np.float32)
    if gray.max() < 10 or int((gray > 50).sum()) < 20:
        return None, None

    rows, cols = np.where(gray > 50)
    crop = gray[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    small = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )

    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = np.asarray(small, dtype=np.float32)

    cy, cx = ndimage.center_of_mass(canvas)
    shift = (int(round(14 - cy)), int(round(14 - cx)))
    canvas = ndimage.shift(canvas, shift, order=1, cval=0)

    preview = np.clip(canvas, 0, 255).astype(np.uint8)
    x = (preview.astype(np.float32) / 255.0 - mean) / (std + 1e-7)
    return x.reshape(1, 28, 28, 1).astype(np.float32), preview


def main():
    import pandas as pd
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas

    st.set_page_config(
        page_title="MNIST Digit Recognizer", page_icon="✏️", layout="centered"
    )

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #fffaf5 0%, #f4f6ff 100%); }
        .block-container { max-width: 900px; padding-top: 3rem; padding-bottom: 3rem; }
        h1 { color: #31204f !important; font-size: 3.25rem !important; font-weight: 800 !important; }
        h2, h3 { color: #31204f !important; }
        .tagline { color: #334155 !important; font-size: 1.2rem; margin-top: 0.35rem; }
        [data-testid="stAlert"] p, [data-testid="stAlert"] div { color: #172554 !important; }
        [data-testid="stCaptionContainer"] p { color: #172554 !important; }
        .setup-card, .result-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.25rem 1.5rem; box-shadow: 0 8px 24px rgba(49, 32, 79, 0.08); }
        .setup-card h3 { color: #31204f; margin-top: 0; }
        .setup-card p, .setup-card li { color: #334155; }
        .result-card { text-align: center; margin-bottom: 1rem; }
        .result-digit { font-size: 96px; font-weight: 800; line-height: 1.1; color: #31204f; }
        .result-conf { font-size: 1.2rem; color: #334155; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_artifacts():
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH)
        params = json.loads(PREPROCESS_PATH.read_text())
        return model, float(params["mean"]), float(params["std"])

    st.title("✏️ MNIST Digit Recognizer")
    st.markdown(
        '<p class="tagline">Draw a digit with your mouse and let an ensemble of CNNs guess it.</p>',
        unsafe_allow_html=True,
    )

    # Check outside the cache so a model exported later is picked up on refresh.
    artifacts = load_artifacts() if MODEL_PATH.exists() and PREPROCESS_PATH.exists() else None
    if artifacts is None:
        st.markdown(
            """
            <div class="setup-card">
              <h3>One quick setup step</h3>
              <p>The app is ready, but the trained model files have not been created yet.</p>
              <ol>
                <li>Open a terminal in this project folder.</li>
                <li>Activate your environment: <code>conda activate mnist-sub10</code></li>
                <li>Train the ensemble once, then refresh this page.</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code("python sub10.py ensemble runs/", language="bash")
        st.caption(
            "This creates mnist_ensemble.keras and preprocess.json next to app.py."
        )
        st.stop()

    model, mean, std = artifacts

    left, right = st.columns(2)
    with left:
        stroke_width = st.slider("Brush size", 8, 30, 18)
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            return_image_data=True,
            key="canvas",
        )
        st.caption("Draw one digit (0–9). Use the trash icon under the canvas to clear.")

    x = preview = None
    if canvas_result is not None and canvas_result.image_data is not None:
        x, preview = preprocess_canvas(canvas_result.image_data.astype(np.uint8), mean, std)

    with right:
        if x is None:
            st.markdown(
                '<div class="result-card"><div class="result-digit">?</div>'
                '<div class="result-conf">Draw a digit to see the prediction</div></div>',
                unsafe_allow_html=True,
            )
        else:
            probs = model(x, training=False).numpy()[0]
            digit = int(np.argmax(probs))
            confidence = float(probs[digit])
            st.markdown(
                f'<div class="result-card"><div class="result-digit">{digit}</div>'
                f'<div class="result-conf">Confidence: {confidence * 100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
            if confidence < 0.6:
                st.warning("Not sure about this one. Try redrawing the digit larger and clearer.")
            st.bar_chart(pd.DataFrame({"probability": probs}, index=range(10)))
            # Inline PNG instead of st.image: avoids Streamlit's /media endpoint.
            buf = io.BytesIO()
            Image.fromarray(preview).resize((112, 112), Image.NEAREST).save(buf, format="PNG")
            st.markdown(
                f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
                'width="112" style="image-rendering: pixelated; border-radius: 6px;">'
                '<p style="color:#64748b; font-size:0.85rem;">What the model sees (28×28)</p>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
