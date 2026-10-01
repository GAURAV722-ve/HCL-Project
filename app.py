import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Student AI Chatbot",
    page_icon="🎓"
)


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Keep only letters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    with open("student_chatbot.pkl", "rb") as file:
        chatbot = pickle.load(file)

    return chatbot


# Try loading model
try:

    chatbot = load_model()

    tfidf = chatbot["tfidf"]
    model = chatbot["model"]

    retrieval_tfidf = chatbot["final_retrieval_tfidf"]
    message_matrix = chatbot["final_message_matrix"]
    rows = chatbot["final_rows"]

    model_loaded = True

except Exception as e:

    model_loaded = False
    error = str(e)


# --------------------------------------------------
# CHATBOT FUNCTION
# --------------------------------------------------

def chatbot_response(question):

    # Check empty question
    if not isinstance(question, str) or not question.strip():

        return {
            "response": "Please enter a question.",
            "intent": "None",
            "confidence": 0.0,
            "similarity": 0.0,
            "matched_question": ""
        }

    # Clean question
    clean_q = preprocess_text(question)

    if not clean_q:
        return {
            "response": "Please enter a meaningful question.",
            "intent": "None",
            "confidence": 0.0,
            "similarity": 0.0,
            "matched_question": ""
        }

    # --------------------------------------------------
    # 1. INTENT PREDICTION
    # --------------------------------------------------

    question_vector = tfidf.transform([clean_q])

    intent = model.predict(question_vector)[0]

    probabilities = model.predict_proba(question_vector)[0]

    confidence = float(probabilities.max())

    # --------------------------------------------------
    # 2. FIND SIMILAR QUESTION
    # --------------------------------------------------

    query_vector = retrieval_tfidf.transform([clean_q])

    similarities = cosine_similarity(
        query_vector,
        message_matrix
    )[0]

    best_index = int(np.argmax(similarities))

    best_similarity = float(similarities[best_index])

    # Get matching row
    best_row = rows.iloc[best_index]

    matched_intent = best_row["Intent"]

    matched_question = best_row["User Message"]

    # --------------------------------------------------
    # 3. DECISION
    # --------------------------------------------------

    if best_similarity >= 0.45:

        response = best_row["Bot Response"]

    elif (
        best_similarity >= 0.30
        and intent == matched_intent
    ):

        response = best_row["Bot Response"]

    else:

        response = (
            "Sorry, I could not find a suitable answer. "
            "Please try asking your question in another way."
        )

    return {
        "response": response,
        "intent": intent,
        "confidence": confidence,
        "similarity": best_similarity,
        "matched_question": matched_question
    }


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🎓 Student Management AI")

st.write(
    "Ask me anything about attendance, registration, "
    "fees, exams, results and other student services."
)


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not model_loaded:

    st.error("❌ Chatbot model could not be loaded.")

    st.code(error)

    st.stop()


st.success("🟢 AI Assistant is ready")


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant" and "details" in message and message["details"]:
            details = message["details"]
            with st.expander("🔎 NLP Details"):
                st.write("**Intent:**", details["intent"])
                st.write("**Confidence:**", f"{details['confidence'] * 100:.2f}%")
                st.write("**Similarity:**", f"{details['similarity'] * 100:.2f}%")
                st.write("**Matched Question:**", details["matched_question"])


# --------------------------------------------------
# USER INPUT & PROCESS
# --------------------------------------------------

question = st.chat_input(
    "Ask your question..."
)

if question:

    # Show and save user message
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Get chatbot answer
    result = chatbot_response(question)

    # Show and save AI response
    with st.chat_message("assistant"):

        st.write(result["response"])

        # Show NLP details
        with st.expander("🔎 NLP Details"):

            st.write(
                "**Intent:**",
                result["intent"]
            )

            st.write(
                "**Confidence:**",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.write(
                "**Similarity:**",
                f"{result['similarity'] * 100:.2f}%"
            )

            st.write(
                "**Matched Question:**",
                result["matched_question"]
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"],
        "details": result
    })


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🎓 Student AI")

    st.write("Student Management Chatbot")

    st.divider()

    st.subheader("Example Questions")

    st.write("• How can I register?")

    st.write("• How can I check my attendance?")

    st.write("• What is the parking policy?")

    st.write("• How can I check my results?")

    st.write("• What is the fee structure?")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()
