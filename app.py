
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
            probability = (
    model.predict_proba(
        vector
    )[0]
)

confidence = (
    max(probability)
    * 100
)
        )
if result == "spam":

    st.error(
        f"🚨 SPAM\n\nConfidence: {confidence:.2f}%"
    )

else:

    st.success(
        f"✅ NOT SPAM\n\nConfidence: {confidence:.2f}%"
    )
        
