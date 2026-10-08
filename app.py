import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack

# ----------------------------------------------------
# 1. Page Configuration & Custom UI Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="Review Sentiment AI | Linear SVM",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
        .main-header {
            font-size: 2.3rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #E2E8F0;
            text-align: center;
        }
        .stAlert {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Text Preprocessing & Artifact Loading
# ----------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_all_artifacts():
    artifacts = {}
    try:
        artifacts['model'] = joblib.load("linear_svm_model.pkl")
        artifacts['encoder'] = joblib.load("label_encoder.pkl")

        # Optional helper artifacts if present
        for name in ['tfidf_vectorizer', 'numeric_scaler', 'feature_columns']:
            try:
                artifacts[name] = joblib.load(f"{name}.pkl")
            except Exception:
                artifacts[name] = None

        return artifacts, True
    except Exception as e:
        return {"error": str(e)}, False

artifacts, is_loaded = load_all_artifacts()

# ----------------------------------------------------
# 3. Model Inference Helper
# ----------------------------------------------------
def predict_dataframe(df_input):
    """
    Handles both direct pipeline input and manual feature matrix generation.
    """
    model = artifacts['model']
    encoder = artifacts['encoder']

    # Ensure standard text and length columns exist
    df_proc = df_input.copy()
    if 'title' not in df_proc.columns:
        df_proc['title'] = ""
    if 'body' not in df_proc.columns:
        df_proc['body'] = df_proc.iloc[:, 0].astype(str)

    df_proc['title'] = df_proc['title'].astype(str)
    df_proc['body'] = df_proc['body'].astype(str)

    df_proc['title_char_len'] = df_proc['title'].str.len()
    df_proc['body_char_len'] = df_proc['body'].str.len()
    df_proc['title_word_count'] = df_proc['title'].apply(lambda x: len(str(x).split()))
    df_proc['body_word_count'] = df_proc['body'].apply(lambda x: len(str(x).split()))

    # 1. Try passing the full feature DataFrame (for ColumnTransformer / Pipeline)
    try:
        preds = model.predict(df_proc)
    except Exception:
        # 2. Try manual stacking if TF-IDF & Scaler are separate
        if artifacts.get('tfidf_vectorizer') is not None and artifacts.get('numeric_scaler') is not None:
            tfidf = artifacts['tfidf_vectorizer']
            scaler = artifacts['numeric_scaler']

            cleaned_bodies = df_proc['body'].apply(clean_text)
            text_vec = tfidf.transform(cleaned_bodies)

            # Scale numeric features
            num_cols = ['title_char_len', 'body_char_len', 'title_word_count', 'body_word_count']
            num_scaled = scaler.transform(df_proc[num_cols])

            X_input = hstack([text_vec, num_scaled])
            preds = model.predict(X_input)
        elif artifacts.get('tfidf_vectorizer') is not None:
            # 3. Fallback: TF-IDF only
            tfidf = artifacts['tfidf_vectorizer']
            cleaned_bodies = df_proc['body'].apply(clean_text)
            X_input = tfidf.transform(cleaned_bodies)
            preds = model.predict(X_input)
        else:
            raise ValueError("Could not match the input format required by the model.")

    # Decode labels
    if hasattr(encoder, 'inverse_transform'):
        labels = encoder.inverse_transform(preds)
    else:
        labels = preds

    return [str(l).capitalize() for l in labels]

# ----------------------------------------------------
# 4. App Header & Sidebar Navigation
# ----------------------------------------------------
st.markdown('<div class="main-header">📱 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify Amazon Mobile Reviews into <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b> using Linear SVM.</div>', unsafe_allow_html=True)

if not is_loaded:
    st.error(f"⚠️ Failed to load model artifacts: {artifacts.get('error')}. Please ensure all `.pkl` files are in the working directory.")
    st.stop()

with st.sidebar:
    st.title("⚙️ Options")
    mode = st.radio("Select Prediction Mode:", ["Single Review", "Bulk Upload (CSV / Excel)"], index=0)
    st.markdown("---")
    st.markdown("### ℹ️ Model Details")
    st.markdown("""
    - **Classifier:** Linear SVM (`LinearSVC`)
    - **Features:** TF-IDF & Engineered Text Features
    - **Classes:** Negative, Neutral, Positive
    """)
    st.markdown("---")
    st.caption("Developed with Streamlit & Scikit-Learn")

# ----------------------------------------------------
# 5. Mode 1: Single Review Prediction
# ----------------------------------------------------
if mode == "Single Review":
    st.subheader("🔍 Single Review Classification")

    with st.container():
        col_title, col_body = st.columns([1, 2])
        with col_title:
            review_title = st.text_input("Review Title", placeholder="e.g., Value for money / Poor battery")
        with col_body:
            review_body = st.text_area("Review Body", placeholder="e.g., The phone's battery life is amazing and camera quality is top-notch!", height=120)

        # Real-time feature preview metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Title Chars", len(review_title))
        c2.metric("Title Words", len(review_title.split()))
        c3.metric("Body Chars", len(review_body))
        c4.metric("Body Words", len(review_body.split()))

        if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
            if not review_body.strip() and not review_title.strip():
                st.warning("Please enter review title or body text.")
            else:
                sample_df = pd.DataFrame({
                    'title': [review_title],
                    'body': [review_body]
                })

                with st.spinner("Analyzing text..."):
                    prediction = predict_dataframe(sample_df)[0]

                st.markdown("### Result")
                if prediction.lower() == "positive":
                    st.success(f"### 🎉 Sentiment: **{prediction}**")
                elif prediction.lower() == "negative":
                    st.error(f"### ⚠️ Sentiment: **{prediction}**")
                else:
                    st.warning(f"### ⚖️ Sentiment: **{prediction}**")

# ----------------------------------------------------
# 6. Mode 2: Bulk CSV / Excel File Upload
# ----------------------------------------------------
else:
    st.subheader("📁 Bulk Review Analysis")
    st.markdown("Upload a CSV or Excel file containing a column of customer reviews (e.g., `body`, `review`, or `text`).")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write(f"**Loaded {len(df_upload)} rows.** Preview of raw data:")
            st.dataframe(df_upload.head(4), use_container_width=True)

            # Select target review column
            col1, col2 = st.columns(2)
            with col1:
                title_col = st.selectbox("Select Title Column (Optional)", ["None"] + list(df_upload.columns))
            with col2:
                # Default to 'body' if present, otherwise first object column
                default_idx = 0
                for idx, c in enumerate(df_upload.columns):
                    if c.lower() in ['body', 'review', 'text', 'comment']:
                        default_idx = idx
                        break
                body_col = st.selectbox("Select Review Body Column", list(df_upload.columns), index=default_idx)

            if st.button("📊 Process All Reviews", type="primary"):
                with st.spinner(f"Classifying {len(df_upload)} reviews..."):
                    eval_df = pd.DataFrame()
                    eval_df['title'] = df_upload[title_col] if title_col != "None" else ""
                    eval_df['body'] = df_upload[body_col]

                    df_upload['Predicted_Sentiment'] = predict_dataframe(eval_df)

                st.success("✅ Classification completed!")
                st.divider()

                # Metrics Overview
                counts = df_upload['Predicted_Sentiment'].value_counts()
                total = len(df_upload)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Reviews", total)
                m2.metric("Positive 😊", counts.get("Positive", 0), f"{(counts.get('Positive', 0)/total)*100:.1f}%")
                m3.metric("Neutral 😐", counts.get("Neutral", 0), f"{(counts.get('Neutral', 0)/total)*100:.1f}%")
                m4.metric("Negative 😡", counts.get("Negative", 0), f"{(counts.get('Negative', 0)/total)*100:.1f}%")

                # Chart Distribution
                st.markdown("#### Sentiment Distribution")
                st.bar_chart(counts)

                # Detailed Data Table
                st.markdown("#### Detailed Results")
                st.dataframe(df_upload, use_container_width=True)

                # Export Button
                csv_data = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Classified Results as CSV",
                    data=csv_data,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv"
                )

        except Exception as err:
            st.error(f"Error processing file: {err}")
