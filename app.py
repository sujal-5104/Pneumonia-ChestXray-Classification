
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, UnidentifiedImageError
from pathlib import Path


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PneumoAI | Chest X-Ray Classification",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROFESSIONAL CSS THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #F3F6FC;
    color: #172B4D;
}

[data-testid="stAppViewContainer"] {
    background-color: #F3F6FC;
}

.stApp p,
.stApp li,
.stApp label {
    color: #24364B;
}

.hero {
    background: linear-gradient(135deg, #102A43, #075985);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 36px;
    font-weight: 800;
}

.hero p {
    color: #E2F3FF !important;
    font-size: 16px;
}

.section-title {
    color: #102A43 !important;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 14px;
}

[data-testid="stSidebar"] {
    background-color: #E5EDF7;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #102A43 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #24364B !important;
}

[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1px solid #CBD8E6;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #CBD8E6;
    border-radius: 14px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #42566D !important;
}

[data-testid="stMetricValue"] {
    color: #102A43 !important;
    font-weight: 800;
}

.stButton > button[kind="primary"] {
    background-color: #075985;
    color: #FFFFFF !important;
    border: 1px solid #075985;
    border-radius: 10px;
    font-weight: 700;
    padding: 12px;
}

.stButton > button[kind="primary"]:hover {
    background-color: #102A43;
    color: #FFFFFF !important;
}

.stProgress > div > div > div {
    background-color: #087F8C;
}

[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #CBD8E6;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #102A43 !important;
    font-weight: 700;
}

[data-testid="stCaptionContainer"] {
    color: #52677E !important;
}

.footer {
    text-align: center;
    color: #52677E !important;
    font-size: 13px;
    padding: 25px 0;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

MODEL_PATH = Path(__file__).parent / "pneumonia_model.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found.")
    st.write("Expected location:")
    st.code(str(MODEL_PATH))
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error("Could not load the trained model.")
    st.code(str(error))
    st.stop()


# =====================================================
# 4. SIDEBAR
# =====================================================

with st.sidebar:
    st.title("🩻 PneumoAI")
    st.caption("Chest X-Ray Classification")

    st.divider()

    st.subheader("About the Project")
    st.write(
        "This educational application uses a "
        "Convolutional Neural Network (CNN) to classify "
        "chest X-ray images."
    )

    st.subheader("How to Use")
    st.markdown("""
    1. Upload an X-ray image.
    2. Preview the image.
    3. Click Analyze X-Ray.
    4. Review the model output.
    """)

    st.divider()
    st.caption("Built with Python, TensorFlow and Streamlit.")


# =====================================================
# 5. MAIN HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>Chest X-Ray Analysis</h1>
    <p>
        Deep Learning Based Image Classification
        <br>
        Upload an image to see the output of your trained CNN.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. UPLOAD AND PREVIEW
# =====================================================

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown(
        '<div class="section-title">Upload X-Ray</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"]
    )

    st.caption("Supported formats: JPG, JPEG and PNG.")

with right_col:
    st.markdown(
        '<div class="section-title">Image Preview</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is None:
        st.info("Your uploaded image will appear here.")


# =====================================================
# 7. PREDICTION
# =====================================================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")

        with right_col:
            st.image(
                image,
                caption="Uploaded Chest X-Ray",
                use_container_width=True
            )

        st.divider()

        if st.button(
            "🔍 Analyze X-Ray",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Analyzing image..."):

                # Match the training input dimensions and grayscale mode.
                resized_image = image.resize((180, 180))

                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32
                )

                # Expected input shape: (1, 180, 180, 1)
                image_array = np.expand_dims(
                    image_array,
                    axis=(0, -1)
                )

                raw_output = model.predict(
                    image_array,
                    verbose=0
                )

                pneumonia_score = float(
                    np.asarray(raw_output).reshape(-1)[0]
                )

                pneumonia_score = float(
                    np.clip(pneumonia_score, 0.0, 1.0)
                )

            # This logic assumes training used:
            # NORMAL = 0 and PNEUMONIA = 1,
            # with a sigmoid output and threshold of 0.5.
            if pneumonia_score >= 0.5:
                predicted_class = "PNEUMONIA"
                class_score = pneumonia_score
            else:
                predicted_class = "NORMAL"
                class_score = 1.0 - pneumonia_score

            # Print diagnostic information in the terminal.
            print("Raw pneumonia score:", pneumonia_score)
            print("Predicted class:", predicted_class)

            # =============================================
            # 8. RESULTS
            # =============================================

            st.markdown(
                '<div class="section-title">Analysis Results</div>',
                unsafe_allow_html=True
            )

            result_col, score_col = st.columns(2)

            with result_col:
                if predicted_class == "NORMAL":
                    st.success("MODEL OUTPUT: NORMAL")
                else:
                    st.error("MODEL OUTPUT: PNEUMONIA")

            with score_col:
                st.metric(
                    "Score for predicted class",
                    f"{class_score * 100:.2f}%"
                )

            st.progress(class_score)

            st.caption(
                "The score is derived from the model output. "
                "It is not a clinically validated probability "
                "or measure of diagnostic certainty."
            )

            # =============================================
            # 9. TECHNICAL DETAILS
            # =============================================

            with st.expander("View Technical Details"):
                st.write("**Model:** CNN")
                st.write("**Input size:** 180 × 180 pixels")
                st.write("**Color mode:** Grayscale")
                st.write("**Threshold:** 0.50")
                st.write(
                    f"**Raw pneumonia score:** {pneumonia_score:.4f}"
                )

            # =============================================
            # 10. MEDICAL DISCLAIMER
            # =============================================

            st.warning(
                "Educational use only. This application is "
                "not a medical diagnostic system and may "
                "produce incorrect predictions. Do not use "
                "its output to make healthcare decisions. "
                "Consult a qualified healthcare professional "
                "for medical concerns."
            )

    except (UnidentifiedImageError, OSError):
        st.error(
            "The uploaded file could not be read. "
            "Please upload a valid JPG, JPEG or PNG image."
        )

    except Exception as error:
        st.error("An error occurred during prediction.")
        st.code(str(error))


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <hr>
    PneumoAI | Deep Learning Project
    <br>
    Educational Demonstration Only
</div>
""", unsafe_allow_html=True)
