import streamlit as st
import string

st.title("CV ↔ Job Description Matcher (MVP)")

def extract_keywords(text):
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # split into list of words
    words = text.split()
    # remove v. common words
    stopwords = {"the","and","to","a","of","in","on","for","with","is","it","as","an","by","at"}
    keywords = [w for w in words if w not in stopwords]
    return set(keywords)

# --- UI ---
cv_text = st.text_area("Paste your CV text")
jd_text = st.text_area("Paste the Job Description")

if st.button("Analyze"):
    if not cv_text or not jd_text:
        st.warning("Please paste both CV and JD.")
    else:
        cv_keywords = extract_keywords(cv_text)
        jd_keywords = extract_keywords(jd_text)

        overlap = cv_keywords & jd_keywords
        match = round(100 * len(overlap) / max(1, len(jd_keywords)), 1)

        st.metric("Match %", f"{match}%")
        st.write("✅ Matched terms:", ", ".join(sorted(list(overlap))[:50]))
        st.write("❌ Missing terms:", ", ".join(sorted(list(jd_keywords - cv_keywords))[:50]))
