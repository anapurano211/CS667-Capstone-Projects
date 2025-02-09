import requests
import hashlib
import hmac
import time
import base64
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from urllib.parse import urlparse

# === Moz API Credentials ===
MOZ_ACCESS_ID = "your_moz_access_id"  # Replace with your Moz Access ID
MOZ_SECRET_KEY = "your_moz_secret_key"  # Replace with your Moz Secret Key

def get_moz_domain_authority(url: str) -> int:
    """
    Fetches the Domain Authority (DA) score from Moz API.
    
    Args:
        url (str): The website URL to check.
        
    Returns:
        int: Domain Authority score (0-100).
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower().replace('www.', '')  # Extract domain

    # Generate Moz API authentication signature
    expires = int(time.time()) + 300  # 5-minute validity
    string_to_sign = f"{MOZ_ACCESS_ID}\n{expires}"
    signature = hmac.new(MOZ_SECRET_KEY.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha1)
    encoded_signature = base64.b64encode(signature.digest()).decode('utf-8')

    # Moz API endpoint
    api_url = f"https://lsapi.seomoz.com/v2/url_metrics"

    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{MOZ_ACCESS_ID}:{MOZ_SECRET_KEY}'.encode()).decode()}",
        "Content-Type": "application/json"
    }

    payload = {
        "targets": [f"https://{domain}"],
        "metrics": ["domain_authority"]
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        data = response.json()

        if "results" in data and data["results"]:
            return int(data["results"][0].get("domain_authority", 50))  # Default to 50 if not found
    except Exception as e:
        print(f"Error fetching Moz DA: {str(e)}")
    
    return 50  # Default fallback value

def rate_url_validity(user_query: str, url: str) -> dict:
    """
    Evaluates the validity of a given URL by computing various metrics including 
    domain trust, content relevance, fact-checking, bias, and citation scores.

    Args:
        user_query (str): The user's original query.
        url (str): The URL to analyze.

    Returns:
        dict: A dictionary containing the final validity score and an explanation.
    """

    # === Step 1: Fetch Page Content ===
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = " ".join([p.text for p in soup.find_all("p")])  # Extract paragraph text
    except Exception as e:
        return {"error": f"Failed to fetch content: {str(e)}"}

    # === Step 2: Domain Authority Check (Moz API) ===
    domain_trust = get_moz_domain_authority(url)

    # === Step 3: Content Relevance (Semantic Similarity using Hugging Face) ===
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    similarity_score = util.pytorch_cos_sim(model.encode(user_query), model.encode(page_text)).item() * 100

    # === Step 4: Fact-Checking (Google Fact Check API) ===
    fact_check_score = check_facts(page_text)

    # === Step 5: Bias Detection (NLP Sentiment Analysis) ===
    sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    sentiment_result = sentiment_pipeline(page_text[:512])[0]  # Process first 512 characters
    bias_score = 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    # === Step 6: Citation Check (Google Scholar via SerpAPI) ===
    citation_count = check_google_scholar(url)
    citation_score = min(citation_count * 10, 100)  # Normalize

    # === Step 7: Compute Final Validity Score ===
    final_score = (
        (0.3 * domain_trust) +
        (0.3 * similarity_score) +
        (0.2 * fact_check_score) +
        (0.1 * bias_score) +
        (0.1 * citation_score)
    )

    # Ensure final score is within 0-100 range
    final_score = max(0, min(final_score, 100))

    # === Step 8: Generate Explanation in One Line ===
    explanation = (
        f"Final score: {final_score:.2f}, Explanation: "
        f"The website has a domain authority of {domain_trust}, indicating a {'highly' if domain_trust > 70 else 'moderately' if domain_trust > 50 else 'low'} reputable source. "
        f"The content relevance score is {similarity_score:.2f}, suggesting the article is "
        f"{'highly' if similarity_score > 70 else 'moderately' if similarity_score > 50 else 'weakly'} relevant. "
        f"Fact-checking sources give it a reliability score of {fact_check_score}. "
        f"The sentiment analysis detected a {sentiment_result['label']} tone, leading to a bias score of {bias_score}. "
        f"The article has {citation_count} citations, contributing a citation score of {citation_score}. "
        f"Overall, the source is considered "
        f"{'highly reliable' if final_score > 80 else 'moderately reliable' if final_score > 60 else 'questionable'}."
    )

    return {
        "Final Validity Score": final_score,
        "Explanation": explanation
    }

# === Helper Function: Fact-Checking via Google API ===
def check_facts(text: str) -> int:
    """
    Cross-checks text against Google Fact Check API.
    Returns a score between 0-100 indicating factual reliability.
    """
    api_url = f"https://toolbox.google.com/factcheck/api/v1/claimsearch?query={text[:200]}"
    try:
        response = requests.get(api_url)
        data = response.json()
        if "claims" in data and data["claims"]:
            return 80  # If found in fact-checking database
        return 40  # No verification found
    except:
        return 50  # Default uncertainty score

# === Helper Function: Citation Count via Google Scholar API ===
def check_google_scholar(url: str) -> int:
    """
    Checks Google Scholar citations using SerpAPI.
    Returns the count of citations found.
    """
    serpapi_key = "efc206f678b71a9a7ec21e5000e5e4b03cbe155fc5716b04f266bc3b0744b595"  # Replace with your SerpAPI key
    params = {"q": url, "engine": "google_scholar", "api_key": serpapi_key}
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        return len(data.get("organic_results", []))
    except:
        return 0  # Assume no citations found

# === Example Usage ===
user_prompt = "I want to learn math for machine learning, can you recommend a site where i can learn?"
url_to_check = "https://www.geeksforgeeks.org/machine-learning-mathematics/"

result = rate_url_validity(user_prompt, url_to_check)
print(result["Explanation"])