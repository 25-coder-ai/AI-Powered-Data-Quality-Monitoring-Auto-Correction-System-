import streamlit as st
import pandas as pd
from utils import check_missing, detect_outliers, quality_score
from utils import overall_quality
from utils import auto_fix


st.title("AI Data Quality Monitoring System")

uploaded_file = st.file_uploader("Upload your dataset")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    st.subheader("Missing Values")
    result = check_missing(data)
    st.write(result)

    if data.isnull().mean().mean() > 0.1:
        st.warning("⚠️ High missing data!")
    st.subheader("Missing Values Chart")
    st.bar_chart(data.isnull().sum())

    st.subheader("Outliers")
    outliers = detect_outliers(data)
    st.write(outliers)

    scores = {k: round(v * 100, 2) for k, v in overall_quality(data).items()}

    st.subheader("Data Quality Dimensions")
    st.write(scores)
    st.bar_chart(scores)

    for key, value in scores.items():
        if value < 80:
            st.warning(f"{key} is low ⚠️")

    st.subheader("Data Quality Score")
    st.success(f"{quality_score(data)} / 100")

    st.subheader("Auto Fix Data")

    if st.button("Fix Data Automatically"):
        fixed_data = auto_fix(data)

        st.success("✅ Data cleaned successfully!")

        st.subheader("Cleaned Data Preview")
        st.write(fixed_data.head())

        st.subheader("New Quality Score")
        st.success(f"{quality_score(fixed_data)} / 100")

        csv = fixed_data.to_csv(index=False)

        st.download_button(
            label="Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )