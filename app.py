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


BRUSH_SIZE = 27  # scaled with the 420 px canvas so strokes match MNIST thickness after resizing
CANVAS_SIZE = 420
ACCENT, ACCENT_SOFT, INK = "#8b5cf6", "#ddd6fe", "#31204f"


def png_data_uri(array, size):
    """Inline PNG (data URI) so images don't go through Streamlit's /media endpoint."""
    buf = io.BytesIO()
    Image.fromarray(array).resize(size, Image.NEAREST).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colorize(fmap):
    """Map a 2-D activation to the app palette: near-white -> violet -> deep purple."""
    lo, hi = float(fmap.min()), float(fmap.max())
    t = (fmap - lo) / (hi - lo) if hi > lo else np.zeros_like(fmap)
    stops = np.array([[248, 247, 255], [139, 92, 246], [49, 32, 79]], dtype=np.float32)
    t2 = np.clip(t * 2, 0, 1)[..., None]
    rgb = np.where((t < 0.5)[..., None], stops[0] + (stops[1] - stops[0]) * t2,
                   stops[1] + (stops[2] - stops[1]) * (np.clip(t * 2 - 1, 0, 1)[..., None]))
    return rgb.astype(np.uint8)


def probability_bars(probs, digit):
    """Fixed-scale 0-1 bar chart as HTML; the predicted digit is highlighted."""
    cols = []
    for i, p in enumerate(probs):
        color = ACCENT if digit == i else ACCENT_SOFT
        label = f"{p * 100:.0f}%" if p >= 0.02 else ""
        cols.append(
            f'<div class="bar-col"><div class="bar-val">{label}</div>'
            f'<div class="bar-track"><div class="bar" style="height:{p * 100:.1f}%; background:{color};"></div></div>'
            f'<div class="bar-label{" bar-label-on" if digit == i else ""}">{i}</div></div>'
        )
    return f'<div class="chart-card"><div class="chart-title">Probability per digit</div><div class="bars">{"".join(cols)}</div></div>'


def pick_member(model):
    """Member shown in the 'inside the model' view: a VGG one if present (simplest flow)."""
    import tensorflow as tf

    members = [l for l in model.layers if isinstance(l, tf.keras.Model)] or [model]
    return next((m for m in members if m.name.startswith("vgg")), members[0]), members


def stage_layers(member):
    """First conv, then the last conv at each spatial resolution."""
    import tensorflow as tf

    convs = [l for l in member.layers if isinstance(l, tf.keras.layers.Conv2D)]
    stages, seen = [convs[0]], {}
    for l in convs[1:]:
        seen[l.output.shape[1]] = l
    stages += [l for l in seen.values() if l is not convs[0]]
    return stages


def stage_name(layer):
    k = layer.kernel_size[0]
    stride = layer.strides[0]
    return f"Conv {k}×{k}" + (f" /{stride}" if stride > 1 else "")


def main():
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
        .tagline { color: #334155 !important; font-size: 1.2rem; margin-top: 0.35rem; margin-bottom: 2.2rem; }
        [data-testid="stAlert"] p, [data-testid="stAlert"] div { color: #172554 !important; }
        [data-testid="stCaptionContainer"] p { color: #172554 !important; }
        .setup-card, .result-card, .chart-card, .stage-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.25rem 1.5rem; box-shadow: 0 8px 24px rgba(49, 32, 79, 0.08); }
        .setup-card h3 { color: #31204f; margin-top: 0; }
        .setup-card p, .setup-card li { color: #334155; }
        .result-card { text-align: center; margin-bottom: 1rem; }
        .result-digit { font-size: 96px; font-weight: 800; line-height: 1.1; color: #31204f; }
        .result-digit.empty { color: #cbd5e1; }
        .result-conf { font-size: 1.2rem; color: #334155; }
        .chart-title, .stage-title { color: #31204f; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.6rem; }
        .bars { display: flex; align-items: flex-end; gap: 6px; height: 150px; }
        [class*="st-key-canvas_"] { width: 100% !important; display: flex; justify-content: center; }
        [class*="st-key-canvas_"] .stBidiComponent { width: 420px !important; margin: 0 auto; }
        .st-key-clear_btn { max-width: 420px; margin: 0 auto; }
        .bar-col { flex: 1; display: flex; flex-direction: column; align-items: center; height: 100%; }
        .bar-val { font-size: 0.7rem; color: #6d28d9; height: 1rem; }
        .bar-track { flex: 1; width: 100%; display: flex; align-items: flex-end; background: #f5f3ff; border-radius: 8px; }
        .bar { width: 100%; border-radius: 8px; min-height: 2px; transition: height 0.25s ease; }
        .bar-label { margin-top: 4px; color: #64748b; font-weight: 600; }
        .bar-label-on { color: #31204f; font-weight: 800; }
        .stage-card { margin-bottom: 0.8rem; }
        .stage-sub { color: #64748b; font-size: 0.8rem; margin: -0.4rem 0 0.6rem; }
        .tiles { display: flex; flex-wrap: wrap; gap: 6px; }
        .tiles img { border-radius: 6px; image-rendering: pixelated; border: 1px solid #ede9fe; }
        .flow { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 0.4rem 0 1rem; }
        .chip { background: #f5f3ff; color: #31204f; border: 1px solid #ddd6fe; border-radius: 999px; padding: 0.2rem 0.7rem; font-size: 0.8rem; font-weight: 600; }
        .arrow { color: #a78bfa; }
        div.stButton > button { border-radius: 999px; border: 1px solid #ddd6fe; color: #31204f; background: #ffffff; font-weight: 600; }
        div.stButton > button:hover { border-color: #8b5cf6; color: #6d28d9; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_artifacts():
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH)
        params = json.loads(PREPROCESS_PATH.read_text())
        member, members = pick_member(model)
        stages = stage_layers(member)
        probe = tf.keras.Model(member.inputs, [l.output for l in stages])
        return model, float(params["mean"]), float(params["std"]), member, members, stages, probe

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

    model, mean, std, member, members, stages, probe = artifacts

    # A new canvas key is the reliable way to wipe the drawing.
    st.session_state.setdefault("canvas_id", 0)

    left, right = st.columns(2)
    with left:
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=BRUSH_SIZE,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            drawing_mode="freedraw",
            return_image_data=True,
            key=f"canvas_{st.session_state.canvas_id}",
        )
        if st.button("🗑️  Clear canvas", key="clear_btn", use_container_width=True):
            st.session_state.canvas_id += 1
            st.rerun()
        st.caption("Draw one digit (0–9), as large as the box.")

    x = preview = None
    if canvas_result is not None and canvas_result.image_data is not None:
        x, preview = preprocess_canvas(canvas_result.image_data.astype(np.uint8), mean, std)

    probs, digit = np.zeros(10, dtype=np.float32), None
    if x is not None:
        probs = model(x, training=False).numpy()[0]
        digit = int(np.argmax(probs))

    with right:
        if digit is None:
            st.markdown(
                '<div class="result-card"><div class="result-digit empty">?</div>'
                '<div class="result-conf">Draw a digit to see the prediction</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-card"><div class="result-digit">{digit}</div>'
                f'<div class="result-conf">Confidence: {probs[digit] * 100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(probability_bars(probs, digit), unsafe_allow_html=True)
        if digit is not None and probs[digit] < 0.6:
            st.warning("Not sure about this one. Try redrawing the digit larger and clearer.")

    st.markdown("### 🔍 Inside the model")
    names = ", ".join(m.name for m in members)
    st.markdown(
        f'<p class="tagline" style="font-size:1rem">The prediction is the average of '
        f'<b>{len(members)} CNNs</b> ({names}). Below, one of them (<b>{member.name}</b>) '
        f'is followed stage by stage: each tile is one feature map, darker = stronger activation.</p>',
        unsafe_allow_html=True,
    )
    flow = ['<span class="chip">Input 28×28×1</span>']
    for l in stages:
        _, h, w, c = l.output.shape
        flow.append(f'<span class="arrow">→</span><span class="chip">{stage_name(l)} · {h}×{w}×{c}</span>')
    flow.append(f'<span class="arrow">→</span><span class="chip">Dense → 10 probs</span>'
                f'<span class="arrow">→</span><span class="chip">Average of {len(members)}</span>')
    st.markdown(f'<div class="flow">{"".join(flow)}</div>', unsafe_allow_html=True)

    if x is None:
        st.caption("Draw a digit to see how it flows through the network.")
    else:
        uri = png_data_uri(preview, (112, 112))
        st.markdown(
            f'<div class="stage-card"><div class="stage-title">1 · Preprocessed input</div>'
            f'<div class="stage-sub">cropped, scaled to 20 px, centred by mass in 28×28 — exactly like MNIST</div>'
            f'<div class="tiles"><img src="{uri}" width="112"></div></div>',
            unsafe_allow_html=True,
        )
        outputs = probe(x, training=False)
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        for k, (l, out) in enumerate(zip(stages, outputs), start=2):
            fmaps = out.numpy()[0]
            h, w, c = fmaps.shape
            top = np.argsort(fmaps.mean(axis=(0, 1)))[::-1][:8]
            size = (72, 72) if h > 1 else (24, 24)
            tiles = "".join(
                f'<img src="{png_data_uri(colorize(fmaps[..., ch]), size)}" width="{size[0]}" title="channel {ch}">'
                for ch in top
            )
            st.markdown(
                f'<div class="stage-card"><div class="stage-title">{k} · {stage_name(l)} <span style="color:#94a3b8;font-weight:500">({l.name})</span></div>'
                f'<div class="stage-sub">{h}×{w} maps × {c} channels · showing the 8 most active</div>'
                f'<div class="tiles">{tiles}</div></div>',
                unsafe_allow_html=True,
            )
        member_probs = member(x, training=False).numpy()[0]
        st.markdown(
            f'<div class="stage-card"><div class="stage-title">{len(stages) + 2} · Output</div>'
            f'<div class="stage-sub">{member.name} alone says <b>{int(member_probs.argmax())}</b> '
            f'({member_probs.max() * 100:.1f}%); the ensemble average says <b>{digit}</b> '
            f'({probs[digit] * 100:.1f}%)</div></div>',
            unsafe_allow_html=True,
        )

    with st.expander(f"Layer table of {member.name}"):
        rows = [
            {"layer": l.name, "type": type(l).__name__,
             "output shape": str(tuple(l.output.shape[1:])), "params": l.count_params()}
            for l in member.layers
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"{member.count_params():,} parameters in this member; "
                   f"{model.count_params():,} in the whole ensemble.")


if __name__ == "__main__":
    main()
