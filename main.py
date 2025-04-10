import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub 
import tensorflow_text as text
import os
os.environ["TORCH_DISABLE_STREAMLIT_PATCH"] = "1"

model = tf.keras.models.load_model(
    "email_model.keras",
    custom_objects={'KerasLayer': hub.KerasLayer}
)


st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")
st.title("📧 Email Spam Classifier")

email_text = st.text_area("Enter your email text here:", height=300)

if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        
        predictions = model.predict([email_text])
        
        predicted_prob = predictions[0][0]
        predicted_class = 1 if predicted_prob > 0.5 else 0
        
        label = "🟢 Not Spam" if predicted_class == 0 else "🔴 Spam"
        
        st.success(f"Prediction: **{label}**")

        with st.container():
            st.markdown(
                f"""
                <div style="border:1px solid #ccc; border-radius:10px; padding:20px; background-color:#f9f9f9; box-shadow:2px 2px 5px #ccc;">
                    <h4 style="color:#555;">From: someone@example.com</h4>
                    <h4 style="color:#555;">To: you@example.com</h4>
                    <h4 style="color:#555;">Subject: (Your Subject Here)</h4>
                    <hr>
                    <p style="font-size:16px; color:#333;">{email_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
