import streamlit as st
import joblib
import re

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Clean text
def clean(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    return text


# Page
st.set_page_config(
    page_title="Smart Spam Detector",
    page_icon="🛡️"
)

# Title
st.title(
    "🛡️ Smart Spam Detection System"
)

st.write(
    "Detect whether a message is Spam or Not Spam using Machine Learning."
)

# Examples
st.info(
"""
Try examples:

• Congratulations! You won ₹5000

• Free entry in weekly contest
"""
)

# Input
msg = st.text_area(
    "✍️ Enter Message",
    height=150
)


# Predict
if st.button("Predict"):

    if msg:

        cleaned = clean(msg)

        vector = vectorizer.transform(
            [cleaned]
        )

        result = model.predict(
            vector
        )[0]

        probability = (
            model.predict_proba(
                vector
            )[0]
        )

        confidence = (
            max(probability)
            * 100
        )

        st.divider()

        if result == "spam":

            st.error(
                f"""
🚨 SPAM

Confidence:
{confidence:.2f}%
"""
            )

        else:

            st.success(
                f"""
✅ NOT SPAM

Confidence:
{confidence:.2f}%
"""
            )

    else:

        st.warning(
            "Enter a message first."
        )


st.caption(
    "Built using Python • Scikit-learn • Streamlit"
)
