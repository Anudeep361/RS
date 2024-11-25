import streamlit as st
from fashion import recommend  

# Background image styling
st.markdown("""
    <style>
        .stApp {
        background-image: url("https://www.shutterstock.com/image-illustration/clothes-on-grunge-background-shelf-600nw-1867100056.jpg");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)

# Page title
st.title('Smart Shopping: AI-Based Fashion Recommendation Engine for Modern Retail')

# Sidebar with user input
query = st.text_input('What exactly are you looking for:', " ")

# Button to trigger recommendation
if st.button('Recommend'):
    # Call recommend function
    result = recommend(query)
    
    # Check if recommendations are available
    if result:
        st.subheader('Recommended Products:')
        for product in result:
            st.write(f"**{product['product_name']}**")
            st.image(product['image_url'], width=150)
            st.write(f"**Price**: {product['price']}")
            st.write(f"**Description**: {product['description']}")
            st.write('---')  # Add a separator between products
    else:
        st.write("No recommendations found for your query. Please try again with a different search.")
