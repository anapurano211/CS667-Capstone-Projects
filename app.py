import streamlit as st
from dotenv import load_dotenv
from Project1_Deliverable_2 import URLValidator  # Assuming your script is in url_validator.py
import os

# Load environment variables
load_dotenv()

# Set Streamlit page title and description
st.set_page_config(page_title="URL Validator", page_icon="🔍")

# Title and description of the app
st.title("URL Validator")
st.write(
    """
    This application evaluates the credibility of a webpage based on several factors:
    - Domain Trust (Is the source reliable?)
    - Content Relevance (How relevant is the content to your query?)
    - Fact-checking (Has the content been verified?)
    - Bias Detection (Does the content show bias?)
    - Citation Checking (How well is the content supported by citations?)

    Simply enter a **search prompt** and a **URL**, and let the app evaluate the URL's credibility.
    """
)

# Create the input fields
user_query = st.text_input("Enter your search prompt:", "")
url_input = st.text_input("Enter the URL to validate:", "")

# Validate button
if st.button("Validate"):
    if user_query and url_input:
        # Initialize the URLValidator
        validator = URLValidator()

        # Call the rate_url_validity function to get the result
        result = validator.rate_url_validity(user_query, url_input)

        if "error" not in result:
            # Display results
            st.subheader("Result:")
            st.write("### Raw Scores:")
            st.write(result["raw_score"])

            st.write("### Final Score:")
            st.write(f"**{result['stars']['score']} Stars** {result['stars']['icon']}")

            st.write("### Explanation:")
            st.write(result["explanation"])
        else:
            st.error("There was an error processing the request. Please try again.")
    else:
        st.warning("Please enter both a search prompt and a URL to validate.")
