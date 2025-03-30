import streamlit as st
import pandas as pd
import pickle
from surprise import KNNBasic

def get_recommendations(data, customer, top_n, algo):
    # Creating an empty list to store the recommended product ids
    recommendations = []

    # Creating a user-item interactions matrix
    user_item_interactions_matrix = data.pivot(index='customer', columns='product', values='rating')

    # Extracting products the customer hasn't interacted with
    non_interacted_products = user_item_interactions_matrix.loc[customer][user_item_interactions_matrix.loc[customer].isnull()].index.tolist()

    # Predicting ratings for non-interacted products
    for product in non_interacted_products:
        est = algo.predict(customer, product).est
        recommendations.append((product, est))

    # Sorting predictions in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:top_n]

df = pd.read_csv("kaputei_data.csv", encoding="utf-8")


# Load trained model (ensure model is saved and loaded correctly)
with open("model.pkl", "rb") as f:
    sim_item_item_optimized = pickle.load(f)

# Streamlit UI
st.title("Product Recommendation System")

# Customer input
customers = df['customer'].unique().tolist()
selected_customer = st.selectbox("Select customer:", customers)

# Submit button
if st.button("Get Recommendations"):
    recommendations = get_recommendations(df, selected_customer, 5, sim_item_item_optimized)
    
    # Display recommendations
    st.write("### Recommended Products:")
    st.dataframe(pd.DataFrame(recommendations, columns=['Product ID', 'Predicted Rating']))
