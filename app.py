import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

# ----------------------------------------------------
# 0. Safety net: install missing packages at startup
#    The proper fix is a requirements.txt in the repo root; this only keeps the
#    app alive if that file is missing or wasn't picked up.
# ----------------------------------------------------
SKLEARN_PIN = ""  # e.g. "==1.6.1" -> set to the version that created your .pkl files

REQUIRED_PACKAGES = {  # import name -> pip name
    "joblib": "joblib",
    "sklearn": "scikit-learn" + SKLEARN_PIN,
    "nltk": "nltk",
    "openpyxl": "openpyxl",
}


def ensure_packages():
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    commands = [
        [sys.executable, "-m", "pip", "install", "--quiet", *missing],
        ["uv", "pip", "install", "--python", sys.executable, "--quiet", *missing],
    ]
    for cmd in commands:
        try:
            subprocess.check_call(cmd)
            importlib.invalidate_caches()
            return
        except Exception:
            continue
    st.error("Could not install: " + ", ".join(missing) + ". Add them to requirements.txt in the repo root and reboot the app.")
    st.stop()


with st.spinner("Setting up dependencies (first run only)..."):
    ensure_packages()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ----------------------------------------------------
# 1. Page Configuration & Custom UI Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="Review Sentiment AI | Linear SVM",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .main-header { font-size: 2.3rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
        .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
        .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
# LabelEncoder sorts classes alphabetically, so this is the order it produces.
DEFAULT_LABELS = ["negative", "neutral", "positive"]

# ----------------------------------------------------
# 2. Text cleaning -- IDENTICAL to clean_text_pipeline() used in training
# ----------------------------------------------------
NEGATION_WORDS = {"no", "not", "nor", "never", "neither"}


@st.cache_resource
def load_nlp_resources():
    """NLTK stopwords + lemmatizer; downloads the corpora on first run."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")

    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        lemmatizer.lemmatize("test")
    return set(words) - NEGATION_WORDS, lemmatizer


def clean_text_pipeline(text) -> str:
    stop_words, lemmatizer = load_nlp_resources()
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)


# ----------------------------------------------------
# 3. Artifact loading
# ----------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    files = {
        "model": "linear_svm_model.pkl",
        "vectorizer": "tfidf_vectorizer.pkl",
        "encoder": "label_encoder.pkl",
    }
    artifacts, errors = {}, []
    for key, fname in files.items():
        path = BASE_DIR / fname
        if not path.exists():
            errors.append(f"`{fname}` not found next to app.py")
            continue
        try:
            artifacts[key] = joblib.load(path)
        except Exception as exc:  # e.g. scikit-learn version mismatch
            errors.append(f"`{fname}` failed to load: {exc}")
    return artifacts, errors


artifacts, load_errors = load_all_artifacts()

# ----------------------------------------------------
# 4. Inference
# ----------------------------------------------------
def predict_texts(titles, bodies):
    """Return (labels, confidences). Labels are Capitalized; 'Unclassified' if no known words remain."""
    cleaned = [clean_text_pipeline(f"{t} {b}") for t, b in zip(titles, bodies)]
    model, vectorizer = artifacts["model"], artifacts["vectorizer"]

    X = vectorizer.transform(cleaned)
    pred = model.predict(X)

    encoder = artifacts.get("encoder")
    labels = list(encoder.inverse_transform(pred)) if encoder is not None else [DEFAULT_LABELS[int(p)] for p in pred]
    labels = [str(l).capitalize() for l in labels]

    conf = [None] * len(labels)
    if hasattr(model, "predict_proba"):
        conf = list(np.max(model.predict_proba(X), axis=1))
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X)
        if dec.ndim > 1:  # softmax over margins: relative score, not a calibrated probability
            exp = np.exp(dec - dec.max(axis=1, keepdims=True))
            conf = list(np.max(exp / exp.sum(axis=1, keepdims=True), axis=1))

    # No usable vocabulary -> the model would just return its bias class; don't present that as a prediction.
    empty = X.getnnz(axis=1) == 0
    labels = ["Unclassified" if e else l for l, e in zip(labels, empty)]
    conf = [None if e else c for c, e in zip(conf, empty)]
    return labels, conf


# ----------------------------------------------------
# 5. Header & Sidebar
# ----------------------------------------------------
st.markdown('<div class="main-header">📱 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify Amazon Mobile Reviews into <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b> using Linear SVM.</div>', unsafe_allow_html=True)

if "model" not in artifacts or "vectorizer" not in artifacts:
    st.error("⚠️ Failed to load model artifacts:\n\n" + "\n".join(f"- {e}" for e in load_errors)
             + "\n\nPut `linear_svm_model.pkl`, `tfidf_vectorizer.pkl` and `label_encoder.pkl` next to `app.py`.")
    st.stop()
if load_errors:
    st.warning("Some optional artifacts had problems: " + "; ".join(load_errors))

with st.sidebar:
    st.title("⚙️ Options")
    mode = st.radio("Select Prediction Mode:", ["Single Review", "Bulk Upload (CSV / Excel)"], index=0)
    st.markdown("---")
    st.markdown("### ℹ️ Model Details")
    st.markdown("""
    - **Classifier:** Linear SVM (`LinearSVC`)
    - **Features:** TF-IDF (1-2 grams) on cleaned title + body
    - **Preprocessing:** stopword removal (negations kept), lemmatization
    - **Classes:** Negative, Neutral, Positive
    - **Note:** labels were derived from star ratings
    """)
    st.markdown("---")
    st.caption("Developed with Streamlit & Scikit-Learn")

# ----------------------------------------------------
# 6. Single review
# ----------------------------------------------------
if mode == "Single Review":
    st.subheader("🔍 Single Review Classification")

    col_title, col_body = st.columns([1, 2])
    with col_title:
        review_title = st.text_input("Review Title", placeholder="e.g., Value for money / Poor battery")
    with col_body:
        review_body = st.text_area("Review Body", placeholder="e.g., The phone's battery life is amazing and camera quality is top-notch!", height=120)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Title Chars", len(review_title))
    c2.metric("Title Words", len(review_title.split()))
    c3.metric("Body Chars", len(review_body))
    c4.metric("Body Words", len(review_body.split()))

    if st.button("🚀 Analyze Sentiment", type="primary"):
        if not review_body.strip() and not review_title.strip():
            st.warning("Please enter review title or body text.")
        else:
            with st.spinner("Analyzing text..."):
                labels, confs = predict_texts([review_title], [review_body])
            prediction, conf = labels[0], confs[0]

            st.markdown("### Result")
            if prediction == "Positive":
                st.success(f"### 🎉 Sentiment: **{prediction}**")
            elif prediction == "Negative":
                st.error(f"### ⚠️ Sentiment: **{prediction}**")
            elif prediction == "Neutral":
                st.warning(f"### ⚖️ Sentiment: **{prediction}**")
            else:
                st.info("Not enough recognizable words to classify this review.")
            if conf is not None:
                st.caption(f"Relative model confidence: {conf * 100:.1f}% (uncalibrated, based on SVM margins)")

# ----------------------------------------------------
# 7. Bulk upload
# ----------------------------------------------------
else:
    st.subheader("📁 Bulk Review Analysis")
    st.markdown("Upload a CSV or Excel file containing a column of customer reviews (e.g., `body`, `review`, or `text`).")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            df_upload.columns = df_upload.columns.astype(str)

            st.write(f"**Loaded {len(df_upload)} rows.** Preview of raw data:")
            st.dataframe(df_upload.head(4))

            col1, col2 = st.columns(2)
            with col1:
                title_col = st.selectbox("Select Title Column (Optional)", ["None"] + list(df_upload.columns))
            with col2:
                default_idx = 0
                for idx, c in enumerate(df_upload.columns):
                    if c.lower() in ["body", "review", "text", "comment"]:
                        default_idx = idx
                        break
                body_col = st.selectbox("Select Review Body Column", list(df_upload.columns), index=default_idx)

            # Results live in session_state so they survive the rerun triggered by the download button;
            # they are discarded when the file or selected columns change.
            run_key = (uploaded_file.name, uploaded_file.size, title_col, body_col)
            if st.session_state.get("bulk_key") != run_key:
                st.session_state.pop("bulk_result", None)

            if st.button("📊 Process All Reviews", type="primary"):
                with st.spinner(f"Classifying {len(df_upload)} reviews..."):
                    bodies = df_upload[body_col].fillna("").astype(str).tolist()
                    titles = (df_upload[title_col].fillna("").astype(str).tolist()
                              if title_col != "None" else [""] * len(bodies))
                    labels, _ = predict_texts(titles, bodies)
                    result = df_upload.copy()
                    result["Predicted_Sentiment"] = labels
                st.session_state["bulk_result"] = result
                st.session_state["bulk_key"] = run_key

            result = st.session_state.get("bulk_result")
            if result is not None:
                st.success("✅ Classification completed!")
                st.divider()

                counts = result["Predicted_Sentiment"].value_counts()
                total = max(len(result), 1)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Reviews", len(result))
                for metric, name, icon in [(m2, "Positive", "😊"), (m3, "Neutral", "😐"), (m4, "Negative", "😡")]:
                    n = int(counts.get(name, 0))
                    metric.metric(f"{name} {icon}", n, f"{n / total * 100:.1f}%", delta_color="off")
                if counts.get("Unclassified", 0):
                    st.caption(f"{int(counts['Unclassified'])} review(s) had no recognizable words and were left Unclassified.")

                st.markdown("#### Sentiment Distribution")
                st.bar_chart(counts)

                st.markdown("#### Detailed Results")
                st.dataframe(result)

                st.download_button(
                    label="📥 Download Classified Results as CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                )
        except Exception as err:
            st.error(f"Error processing file: {err}")
