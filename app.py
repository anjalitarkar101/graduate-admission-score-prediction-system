# ==========================================================
# app.py - Graduate Admission Score Prediction Web App
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from predict import load_model_and_artifacts, preprocess_user_input, predict_admission_score

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Graduate Admission Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Graduate Admission Score Prediction using ANN")
st.markdown("Enter your profile details to predict your admission chance score!")


# ==========================================================
# Load Model
# ==========================================================
@st.cache_resource
def get_model_and_artifacts():
    """
    Load the trained model, scaler, and feature names.

    Returns:
        model: Trained neural network
        scaler: Fitted MinMaxScaler
        feature_names: List of features in correct order
    """
    try:
        return load_model_and_artifacts()
    except FileNotFoundError:
        return None, None, None


model, scaler, feature_names = get_model_and_artifacts()

if model is None:
    st.error("❌ Model not found! Please run: python train_model.py")
    st.stop()

st.success("✅ Model loaded successfully!")

# ==========================================================
# Sidebar - Instructions
# ==========================================================
with st.sidebar:
    st.header("📖 How It Works")
    st.markdown("""
    1. Enter your academic profile
    2. Click **Predict Admission Score**
    3. Get your **admission chance score**

    **Key Factors:**
    - GRE Score (260-340)
    - TOEFL Score (0-120)
    - CGPA (0-10)
    - Research Experience
    - University Rating
    - SOP & LOR Strength
    """)

    st.divider()

    st.header("📊 Model Information")
    st.markdown(f"""
    - **Model:** ANN (Neural Network)
    - **Input Features:** 7
    - **Output:** Admission Chance Score (0-100%)
    - **Type:** Regression
    """)

# ==========================================================
# User Input Form
# ==========================================================
st.subheader("📝 Enter Your Profile")

col1, col2, col3 = st.columns(3)

with col1:
    # GRE Score (260-340)
    gre_score = st.number_input(
        "GRE Score",
        min_value=260,
        max_value=340,
        value=320,
        help="260-340"
    )

    # TOEFL Score (0-120)
    toefl_score = st.number_input(
        "TOEFL Score",
        min_value=0,
        max_value=120,
        value=110,
        help="0-120"
    )

    # University Rating (1-5)
    university_rating = st.selectbox(
        "University Rating",
        [1, 2, 3, 4, 5],
        help="1-5 (5 being highest)"
    )

with col2:
    # SOP Strength (1-5)
    sop = st.slider(
        "SOP Strength",
        min_value=1.0,
        max_value=5.0,
        value=3.5,
        step=0.5,
        help="1-5"
    )

    # LOR Strength (1-5)
    lor = st.slider(
        "LOR Strength",
        min_value=1.0,
        max_value=5.0,
        value=3.5,
        step=0.5,
        help="1-5"
    )

    # CGPA (0-10)
    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=8.5,
        step=0.1,
        help="0-10"
    )

with col3:
    # Research Experience (0 or 1)
    research = st.selectbox(
        "Research Experience",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    # Display tips based on current inputs
    st.caption("💡 **Tips for Higher Admission:**")
    if research == 1:
        st.info("✅ Research experience increases chances")
    if cgpa > 8.5:
        st.info("✅ High CGPA (8.5+) helps")
    if gre_score > 325:
        st.info("✅ Good GRE score (325+)")
    if toefl_score > 110:
        st.info("✅ Good TOEFL score (110+)")

# Create student data dictionary
student_data = {
    'GRE Score': gre_score,
    'TOEFL Score': toefl_score,
    'University Rating': university_rating,
    'SOP': sop,
    'LOR': lor,
    'CGPA': cgpa,
    'Research': research
}

# ==========================================================
# Predict Button
# ==========================================================
if st.button("🔮 Predict Admission Score", type="primary"):
    with st.spinner("Analyzing your profile..."):
        try:
            # Preprocess user input
            scaled_input = preprocess_user_input(student_data, scaler, feature_names)

            # Make prediction
            admission_chance_score = predict_admission_score(scaled_input, model)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    # ==========================================================
    # Display Results
    # ==========================================================
    st.divider()
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        # Show admission chance with meter
        st.markdown(f"### Admission Chance Score: **{admission_chance_score * 100:.1f}%**")

        # Progress bar
        st.progress(admission_chance_score)

        # Interpretation
        if admission_chance_score >= 0.8:
            st.success("🟢 **High Chance!** You have a strong profile!")
        elif admission_chance_score >= 0.6:
            st.warning("🟡 **Moderate Chance** - Consider improving your profile")
        else:
            st.error("🔴 **Low Chance** - Work on improving your profile")

        # Admission decision
        if admission_chance_score >= 0.7:
            st.balloons()
            st.success("🎉 **Congratulations! You are likely to get admission!**")
        elif admission_chance_score >= 0.5:
            st.info("📝 **Borderline candidate - Strengthen your profile**")
        else:
            st.info("📚 **Consider improving your profile for better chances**")

    with col2:
        # Display student profile
        st.markdown("### 📋 Your Profile")

        profile_data = {
            'GRE Score': gre_score,
            'TOEFL Score': toefl_score,
            'University Rating': university_rating,
            'SOP': sop,
            'LOR': lor,
            'CGPA': cgpa,
            'Research': 'Yes' if research == 1 else 'No'
        }
        profile_df = pd.DataFrame([profile_data])
        st.dataframe(profile_df, use_container_width=True)

    # ==========================================================
    # Detailed Analysis
    # ==========================================================
    st.divider()
    st.subheader("📌 Detailed Analysis")

    # Compare with average
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Your Admission Score", f"{admission_chance_score * 100:.1f}%")

    with col2:
        # Average admission score from dataset
        avg_score = 0.72
        diff = (admission_chance_score - avg_score) * 100
        st.metric(
            "Compared to Average",
            f"{diff:+.1f}%",
            delta_color="normal" if diff > 0 else "inverse"
        )

    with col3:
        risk_level = "Low" if admission_chance_score >= 0.7 else "Moderate" if admission_chance_score >= 0.5 else "High"
        st.metric("Risk Level", risk_level)

    # ==========================================================
    # Improvement Suggestions
    # ==========================================================
    st.subheader("💡 Improvement Suggestions")

    suggestions = []

    # GRE Score suggestions
    if gre_score < 320:
        suggestions.append("📈 **GRE Score** - Consider retaking GRE (target: 320+)")
    elif gre_score < 330:
        suggestions.append("✅ **GRE Score** - Good score")
    else:
        suggestions.append("🌟 **GRE Score** - Excellent score!")

    # TOEFL Score suggestions
    if toefl_score < 100:
        suggestions.append("📈 **TOEFL Score** - Consider retaking TOEFL (target: 100+)")
    elif toefl_score < 110:
        suggestions.append("✅ **TOEFL Score** - Good score")
    else:
        suggestions.append("🌟 **TOEFL Score** - Excellent score!")

    # CGPA suggestions
    if cgpa < 7.5:
        suggestions.append("📈 **CGPA** - Consider improving academic performance")
    elif cgpa < 8.5:
        suggestions.append("✅ **CGPA** - Good CGPA")
    else:
        suggestions.append("🌟 **CGPA** - Excellent CGPA!")

    # Research suggestions
    if research == 0:
        suggestions.append("🔬 **Research** - Add research experience to your profile")
    else:
        suggestions.append("✅ **Research** - Good research experience")

    # SOP suggestions
    if sop < 3.5:
        suggestions.append("📝 **SOP** - Strengthen your Statement of Purpose")
    else:
        suggestions.append("✅ **SOP** - Strong Statement of Purpose")

    # LOR suggestions
    if lor < 3.5:
        suggestions.append("📝 **LOR** - Get stronger Letters of Recommendation")
    else:
        suggestions.append("✅ **LOR** - Strong Letters of Recommendation")

    # Display suggestions in two columns
    col1, col2 = st.columns(2)
    mid = len(suggestions) // 2 + len(suggestions) % 2

    with col1:
        for suggestion in suggestions[:mid]:
            st.write(suggestion)

    with col2:
        for suggestion in suggestions[mid:]:
            st.write(suggestion)

# ==========================================================
# Footer
# ==========================================================
st.markdown("---")
st.caption("🎓 Powered by TensorFlow + Streamlit")