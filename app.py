import streamlit as st
import joblib
import re

# -------------------
# Load files
# -------------------

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -------------------
# Session
# -------------------

if "history" not in st.session_state:
    st.session_state.history = []


# -------------------
# Clean text
# -------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    return text


# -------------------
# Page Config
# -------------------

st.set_page_config(
    page_title="Smart Spam Detector",
    page_icon="🛡️"
)


# -------------------
# Title
# -------------------

st.title(
    "🛡️ Smart Spam Detection System"
)

st.markdown(
"""
Detect whether a message is **Spam** or **Safe**
using Machine Learning.
"""
)


# -------------------
# Examples
# -------------------

st.info(
"""
Try Examples:

• Congratulations! You won ₹5000

• Free entry in weekly contest

"""
)


# -------------------
# Input
# -------------------

msg = st.text_area(
    "✍️ Enter Message",
    height=150
)


# -------------------
# Predict
# -------------------

if st.button("Predict"):

    if msg:

        cleaned = clean_text(
            msg
        )

        vector = vectorizer.transform(
            [cleaned]
        )

        result = model.predict(
            vector
        )[0]

        probability = model.predict_proba(
            vector
        )[0]

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

            st.subheader(
                "Spam Score"
            )

            st.progress(
                int(confidence)
            )

            st.warning(
"""
⚠️ Safety Tips

• Do not click unknown links

• Do not share OTP

• Verify sender details
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

        st.session_state.history.append(
            {
                "Message": msg,
                "Result": result.upper()
            }
        )

    else:

        st.warning(
            "Enter a message first."
        )


# -------------------
# History
# -------------------

if st.session_state.history:

    st.subheader(
        "Prediction History"
    )

    st.table(
        st.session_state.history
    )


# -------------------
# Footer
# -------------------

st.caption(
"""
Built using
Python •
Scikit-learn •
Streamlit
"""
)
