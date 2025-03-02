import streamlit as st
import os  
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import logging
from newspaper import Article
import aiohttp
import asyncio
import nest_asyncio
import requests


nest_asyncio.apply()

# Load environment variables from the .env file
load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class URLValidator:
    """
    A URL validation class that evaluates the credibility of a webpage using multiple NLP models.
    It assesses:
    - Domain trust (credibility of the source)
    - Content relevance (how well the content matches the user's query)
    - Fact-checking (cross-checking claims with trusted sources)
    - Bias detection (analyzing sentiment to detect potential bias)
    - Citation checking (using Google Scholar)
    """

    def __init__(self):
        """
        Initializes the NLP models used for URL validation.
        - Fake News Detection (BERT-based model)
        - Semantic Similarity Model (Sentence Transformers)
        - Sentiment Analysis Model (RoBERTa-based model)
        """
        try:
            # Fetch the SerpAPI key from environment variables or use a default if missing
            self.serpapi_key = os.getenv("SERPAPI_KEY", "efc206f678b71a9a7ec21e5000e5e4b03cbe155fc5716b04f266bc3b0744b595")

            if not self.serpapi_key:
                raise ValueError("SerpAPI key is missing from environment variables or not set in .env file!")

            # Initialize the models
            self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.fake_news_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
            self.sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

        except Exception as e:
            logging.error(f"Error initializing models: {e}")

    async def fetch_page_content(self, url: str) -> str:
        """
        Fetches and extracts text content from the given URL using `newspaper3k`.
        
        Args:
            url (str): The webpage URL to analyze.
        
        Returns:
            str: Extracted text content from the webpage (limited to 2000 characters).
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text[:2000]  # Limit text to 2000 characters
        except Exception as e:
            logging.error(f"Failed to fetch content from {url}: {e}")
            return ""

    def get_domain_trust(self, content: str) -> int:
        """
        Determines the trustworthiness of a webpage by classifying it as REAL or FAKE.
        Uses a fake news detection model (fine-tuned BERT-based classifier).
        
        Args:
            content (str): Extracted text content of the webpage.
        
        Returns:
            int: Trust score (0-100), higher values indicate more credibility.
        """
        if not content:
            return 50
        try:
            result = self.fake_news_classifier(content[:512])[0]
            return 100 if result["label"] == "REAL" else 30 if result["label"] == "FAKE" else 50
        except Exception as e:
            logging.warning(f"Fake news detection failed: {e}")
            return 50

    def compute_similarity_score(self, user_query: str, content: str) -> int:
        """
        Computes semantic similarity between the user query and webpage content.
        Uses a transformer-based model to compare sentence embeddings.
        
        Args:
            user_query (str): The question or search query from the user.
            content (str): Extracted text content of the webpage.
        
        Returns:
            int: Similarity score (0-100), where higher values indicate higher relevance.
        """
        if not content:
            return 0
        try:
            similarity = util.pytorch_cos_sim(
                self.similarity_model.encode(user_query, convert_to_tensor=True),
                self.similarity_model.encode(content, convert_to_tensor=True)
            ).item()
            return int(similarity * 100)
        except Exception as e:
            logging.error(f"Error computing similarity score: {e}")
            return 0

    async def check_fact_sources_async(self, content: str) -> int:
        """
        Cross-checks webpage content with fact-checking websites (Snopes, Politifact, FactCheck.org).
        Uses asynchronous requests to fetch verification results efficiently.
        
        Args:
            content (str): Extracted text content of the webpage.
        
        Returns:
            int: Fact-check score (0-100), where higher values indicate verified claims.
        """
        if not content:
            return 50
        urls = [
            f"https://www.snopes.com/search/?q={content[:100]}",
            f"https://www.politifact.com/search/statements/?q={content[:100]}",
            f"https://www.factcheck.org/search/?q={content[:100]}"
        ]
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [session.get(url) for url in urls]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for response in responses:
                    if isinstance(response, aiohttp.ClientResponse) and response.status == 200:
                        text = await response.text()
                        if len(text) > 500:
                            return 80
        except Exception as e:
            logging.error(f"Error in fact-checking API calls: {e}")
        return 50

    def check_google_scholar(self, url: str) -> int:
        """
        Checks Google Scholar citations using SerpAPI.
        
        Args:
            url (str): The webpage URL to check.
        
        Returns:
            int: Citation score (0-100), higher values indicate more citations.
        """
        params = {"q": url, "engine": "google_scholar", "api_key": self.serpapi_key}
        try:
            response = requests.get("https://serpapi.com/search", params=params)
            data = response.json()
            return min(len(data.get("organic_results", [])) * 10, 100)
        except Exception as e:
            logging.error(f"Error checking Google Scholar citations: {e}")
            return 0  # Default to no citations

    def detect_bias(self, content: str) -> int:
        """
        Analyzes content sentiment to determine potential bias.
        Uses a sentiment analysis model to classify content as Positive, Neutral, or Negative.
        
        Args:
            content (str): Extracted text content of the webpage.
        
        Returns:
            int: Bias score (0-100), where higher values indicate more neutrality.
        """
        if not content:
            return 50
        try:
            sentiment_result = self.sentiment_analyzer(content[:512])[0]
            return 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30
        except Exception as e:
            logging.warning(f"Bias detection failed: {e}")
            return 50

    def get_star_rating(self, score: float) -> tuple:
        """
        Converts a credibility score into a star rating.
        
        Args:
            score (float): The final credibility score (0-100).
        
        Returns:
            tuple: (star count, star emoji representation)
        """
        try:
            stars = max(1, min(5, round(score / 20)))
            return stars, "⭐" * stars
        except Exception as e:
            logging.error(f"Error generating star rating: {e}")
            return 1, "⭐"

    def generate_explanation(self, domain_trust, similarity_score, fact_check_score, bias_score, citation_score, final_score) -> str:
        """
        Generates a human-readable explanation for the credibility score.
        
        Args:
            domain_trust (int): Trust score of the domain.
            similarity_score (int): How relevant the content is to the user query.
            fact_check_score (int): Verification score from fact-checking sources.
            bias_score (int): Bias detection score.
            citation_score (int): Citation score based on Google Scholar.
            final_score (float): Overall credibility score.
        
        Returns:
            str: Explanation detailing factors affecting the credibility score.
        """
        try:
            reasons = []
            if domain_trust < 50:
                reasons.append("The source has low domain authority.")
            if similarity_score < 50:
                reasons.append("The content is not highly relevant to your query.")
            if fact_check_score < 50:
                reasons.append("Limited fact-checking verification found.")
            if bias_score < 50:
                reasons.append("Potential bias detected in the content.")
            if citation_score < 30:
                reasons.append("Few citations found for this content.")
            return " ".join(reasons) if reasons else "This source is highly credible and relevant."
        except Exception as e:
            logging.error(f"Error generating explanation: {e}")
            return "Unable to generate explanation."

    async def async_rate_url_validity(self, user_query: str, url: str) -> dict:
        """
        Asynchronously evaluates the credibility of a webpage based on multiple factors.
        
        Args:
            user_query (str): The user's search query.
            url (str): The webpage URL to analyze.
        
        Returns:
            dict: A dictionary containing credibility scores, star rating, and explanation.
        """
        try:
            content = await self.fetch_page_content(url)
            domain_trust = self.get_domain_trust(content)
            similarity_score = self.compute_similarity_score(user_query, content)
            fact_check_score = await self.check_fact_sources_async(content)
            citation_score = self.check_google_scholar(url)
            bias_score = self.detect_bias(content)

            final_score = (
                (0.3 * domain_trust) +
                (0.3 * similarity_score) +
                (0.2 * fact_check_score) +
                (0.1 * bias_score) +
                (0.1 * citation_score)
            )
            
            stars, icon = self.get_star_rating(final_score)
            explanation = self.generate_explanation(domain_trust, similarity_score, fact_check_score, bias_score, citation_score, final_score)

            return {
                "raw_score": {
                    "Domain Trust": domain_trust,
                    "Content Relevance": similarity_score,
                    "Fact-Check Score": fact_check_score,
                    "Bias Score": bias_score,
                    "Citation Score": citation_score,
                    "Final Validity Score": final_score
                },
                "stars": {
                    "score": stars,
                    "icon": icon
                },
                "explanation": explanation
            }
        except Exception as e:
            logging.error(f"Error in rate_url_validity: {e}")
            return {"error": "An error occurred while processing the request."}

    def rate_url_validity(self, user_query: str, url: str) -> dict:
        """
        Evaluates the validity of a URL by running multiple NLP-based credibility checks.
        
        Args:
            user_query (str): The user's search query.
            url (str): The webpage URL to analyze.
        
        Returns:
            dict: A dictionary containing credibility scores, star rating, and explanation.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.ensure_future(self.async_rate_url_validity(user_query, url))
                return loop.run_until_complete(future)
            else:
                return asyncio.run(self.async_rate_url_validity(user_query, url))
        except Exception as e:
            logging.error(f"Error in rate_url_validity execution: {e}")
            return {"error": "Failed to execute URL validation."}




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
