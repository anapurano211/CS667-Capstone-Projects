# Andrew Napurano - CS667 Report

## Pace University  
**Project #1 – URL Validator using NLP**  

### **Application Description**
The following **URL Validator** is an **NLP-based machine learning model** that evaluates the **credibility of web links** based on several criteria. This allows users to enhance their own search inquiries by **inputting a question** into the model along with the **associated URL** that originally helped answer their question. The URL may be from **Google search results** or even a **LLM chatbot** such as ChatGPT.  

The application then **returns a weighted score** based on how well that URL answered the question using the following parameters and weightings:

### **Scoring Mechanism**
- **Domain Trust (35%)** → Measures website credibility using **fake news detection**
- **Content Relevance (30%)** → Compares webpage text with user query using **NLP similarity**
- **Fact-Check Score (20%)** → Verifies claims via **fact-checking sites** (Snopes, Politifact, etc.)
- **Bias Score (15%)** → Detects **emotional or political bias** using **sentiment analysis**

---

## **NLP Model Information**
**Natural Language Processing (NLP)** is a branch of **Artificial Intelligence (AI)** that allows computers to **understand, interpret, and generate human language**. It enables machines to process text and speech like humans do.

Our application relies on **three NLP-based machine learning models** for **fake news detection, sentiment analysis, and similarity measurement**. This provides a **systematic, cost-effective** approach to aiding users in their search inquiries.

### **Models Used:**
1. **mrm8488/bert-tiny-finetuned-fake-news-detection** (Hugging Face) → Classifies content as **real or fake**  
2. **sentence-transformers/all-mpnet-base-v2** (Sentence Transformers) → Measures how **similar** the webpage content is to the user's query  
3. **cardiffnlp/twitter-roberta-base-sentiment** (Hugging Face) → Detects **emotional/political bias** in content recommended to the user via website URLs  

---

## **Technical Implementation**

### **Class Object**
The application leverages a **`URLValidator` Class** with an **`__init__`** method that acts as the URL validator and **loads all AI model dependencies**. This **reduces redundant API calls** and organizes the code efficiently, allowing it to be updated and manipulated as needed.

Classes take in properties as objects and create the **blueprint** for the rest of the code, which in our case, consists of several **functions that evaluate URL validity**.

### **Code Docstrings**
The **code uses docstrings** in each class and function to help developers **understand what each part of the code does**. This **multiline documentation** is crucial for maintaining **organized model documentation** and for collaboration with other developers.

### **Exception Handling**
The application leverages **`try/except`** blocks to allow the code to **fail gracefully** if an error occurs. Additionally, we added an **enhancement** to **log each error**, allowing users to receive **feedback** on which step the model failed. This provides a **smoother debugging process**.

### **Return Statements**
All functions leverage a **return statement** at the end of each code block to **output the appropriate data type** required for each step. For example, in the **scoring mechanism**, since we require **integers and floats** for calculations, we explicitly specify the return type.

Additionally, **exception handling** is implemented in this step to ensure the model doesn't break unexpectedly.

---

**End of Report**
