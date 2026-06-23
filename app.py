import streamlit as st
import pandas as pd
import joblib
import gzip
import pickle

# ----------------------------
# Load Saved Models
# ----------------------------

kmeans = joblib.load("kmeans.pkl")
scaler = joblib.load("scaler.pkl")

with gzip.open("similarity.pkl.gz", "rb") as f:
    similarity_df = pickle.load(f)

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Product Recommendation"
    ]
)

# ----------------------------
# Home Page
# ----------------------------

if page == "Home":

    st.title("🛒 Shopper Spectrum")

    st.markdown("""
    ## Customer Segmentation and Product Recommendations in E-Commerce

    This application provides:

    ✅ Customer Segmentation using RFM Analysis

    ✅ Product Recommendation using Collaborative Filtering

    ### Modules

    - Customer Segmentation
    - Product Recommendation
    """)

# ----------------------------
# Customer Segmentation
# ----------------------------

elif page == "Customer Segmentation":

    st.title("🎯 Customer Segmentation")

    recency = st.number_input(
        "Recency",
        min_value=0.0,
        value=30.0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0.0,
        value=5.0
    )

    monetary = st.number_input(
        "Monetary Value",
        min_value=0.0,
        value=1000.0
    )

    if st.button("Predict Segment"):

        input_data = scaler.transform(
            [[recency, frequency, monetary]]
        )

        cluster = kmeans.predict(input_data)[0]

        st.success(
            f"Predicted Customer Segment: Cluster {cluster}"
        )

# ----------------------------
# Product Recommendation
# ----------------------------

elif page == "Product Recommendation":

    st.title("🛍 Product Recommendation System")

    product_name = st.text_input(
        "Enter Product Name"
    )

    if st.button("Get Recommendations"):

        if product_name in similarity_df.columns:

            recommendations = (
                similarity_df[product_name]
                .sort_values(ascending=False)
                .iloc[1:6]
            )

            st.success("Top 5 Recommended Products")

            for item in recommendations.index:
                st.write("✅", item)

        else:
            st.error(
                "Product not found. Please enter the exact product name."
            )

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")
st.markdown("Developed for Shopper Spectrum Project")