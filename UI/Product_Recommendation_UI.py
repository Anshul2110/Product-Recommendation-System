
import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Step 1: Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ansul\OneDrive\Desktop\data science project\Product-Recommendation-System\Data\ratings_processed.csv")   # Replace with your file path
    df = df.dropna(subset=["userId", "productId", "Rating"])
    df["userId"] = df["userId"].astype(str)
    df["productId"] = df["productId"].astype(str)
    return df

df = load_data()
st.title(" Amazon Product Recommendation System")


# --- Step 2: Train Models ---
@st.cache_resource
def train_models(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userId", "productId", "Rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Train SVD
    svd = SVD()
    svd.fit(trainset)

    # Train Item-KNN
    sim_options = {"name": "cosine", "user_based": False}
    knn_item = KNNBasic(sim_options=sim_options)
    knn_item.fit(trainset)

    return svd, knn_item

svd, knn_item = train_models(df)

# --- Step 3: Hybrid Recommendation Function ---
def hybrid_recommend(user_id, top_n=5, alpha=0.7):
    all_products = df["productId"].unique()
    user_rated = df[df["userId"] == user_id]["productId"].unique()

    scores = []
    for prod in all_products:
        if prod in user_rated:
            continue
        pred_svd = svd.predict(user_id, prod).est
        pred_knn = knn_item.predict(user_id, prod).est
        final_score = alpha * pred_svd + (1 - alpha) * pred_knn
        scores.append((prod, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return scores

# --- Step 4: Streamlit UI ---
st.sidebar.header("User Options")
user_ids = df["userId"].unique()
selected_user = st.sidebar.selectbox("Select a User", user_ids)

alpha = st.sidebar.slider("Hybrid Weight (SVD vs KNN)", 0.0, 1.0, 0.7, 0.1)
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

if st.sidebar.button("Get Recommendations"):
    recs = hybrid_recommend(selected_user, top_n=top_n, alpha=alpha)
    st.subheader(f"Top {top_n} Recommendations for User {selected_user}")
    recs_df = pd.DataFrame(recs, columns=["ProductID", "Predicted Rating"])
    st.table(recs_df)

# --- Step 5: Popularity-Based Recommendations ---
st.subheader("Top 10 Popular Products (By Avg Rating)")
popularity = df.groupby("productId")["Rating"].mean().sort_values(ascending=False).head(10)
st.table(popularity)
