
import streamlit as st
import joblib
import re

# Load files
model = joblib.load(
    "model.pkl"
)

vectorizer = joblib.load(
    "vectorizer.pkl"
)

# Clean text
def clean(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    return text


# UI
st.title(
    "📩 Spam Message Classifier"
)

st.write(
    "Type a message and predict"
)

msg = st.text_area(
    "Enter Message"
)

if st.button(
    "Predict"
):

    if msg:

        cleaned = clean(msg)

        vector = (
            vectorizer
            .transform(
                [cleaned]
            )
        )

        result = (
            model.predict(
                vector
            )[0]
        )

        if result == "spam":

            st.error(
                "🚨 SPAM"
            )

        else:

            st.success(
                "✅ NOT SPAM"
            )
