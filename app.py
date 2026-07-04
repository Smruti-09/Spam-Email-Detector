import streamlit as st
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# ---------------- SESSION STATE (HISTORY) ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📧 Spam Email Detection System</h1>",
    unsafe_allow_html=True
)

st.write("### Enter an email message and check whether it is Spam or Not")

# ---------------- INPUT ----------------
text = st.text_area("✉️ Email Text", height=150)

# ---------------- BUTTON ----------------
predict_btn = st.button("🚀 Predict")

# ---------------- PREDICTION ----------------
if predict_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        data = vectorizer.transform([text])

        prediction = model.predict(data)

        # probability (check if model supports it)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(data)[0][1]  # spam probability
        else:
            prob = 0

        spam_percent = prob * 100

        st.markdown("---")
        st.subheader("📊 Result")

        if prediction[0] == 1:
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ NOT SPAM EMAIL")

        # progress bar
        st.progress(int(spam_percent))

        st.write(f"🔥 Spam Probability: **{spam_percent:.2f}%**")

        # save to history
        st.session_state.history.append({
            "text": text,
            "result": "SPAM" if prediction[0] == 1 else "NOT SPAM",
            "probability": f"{spam_percent:.2f}%"
        })

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write("Machine Learning Spam Detector")

st.sidebar.markdown("---")
st.sidebar.write("🧠 Model: Logistic Regression / Naive Bayes")
st.sidebar.write("📊 Features: TF-IDF")
st.sidebar.write("📂 Dataset: SMS Spam Dataset")

# ---------------- HISTORY ----------------
st.sidebar.markdown("---")
st.sidebar.title("📜 History")

if st.sidebar.button("🧹 Clear History"):
    st.session_state.history = []

for i, item in enumerate(reversed(st.session_state.history[-10:])):
    st.sidebar.write(f"**{i+1}. {item['result']}**")
    st.sidebar.caption(item["probability"])