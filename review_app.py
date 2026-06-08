import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("Amazone_review_Model.joblib")
vectorizer = joblib.load("Vectorizer.joblib")

# Page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Amazon Review Sentiment Analyzer")
st.write("Enter an Amazon product review and predict its sentiment.")

# User Input
review = st.text_area(
    "Enter Review",
    height=150,
    placeholder="Example: This product is amazing and worth every penny..."
)

# Predict Button
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        # Transform text
        review_vector = vectorizer.transform([review])

        # Prediction
        prediction = model.predict(review_vector)[0]

        # Probability
        probability = model.predict_proba(review_vector)

        confidence = probability.max() * 100

        # Display result
        if prediction == "Positive":
            st.success(
                f"😊 Positive Review\n\nConfidence: {confidence:.2f}%"
            )

        elif prediction == "Negative":
            st.error(
                f"😞 Negative Review\n\nConfidence: {confidence:.2f}%"
            )

        else:
            st.info(
                f"😐 Neutral Review\n\nConfidence: {confidence:.2f}%"
            )

        # Show probabilities
        st.subheader("Prediction Probabilities")

        for label, prob in zip(model.classes_, probability[0]):
            st.write(f"{label}: {prob:.2%}")