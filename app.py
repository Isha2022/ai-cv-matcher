import streamlit as st
import string
import spacy

st.title("CV ↔ Job Description Matcher (MVP)")

# Load spaCy model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")
nlp = load_nlp()

# Custom stopword list for very common, short words
STOP_SHORT = {
    "the","and","to","a","of","in","on","for","with","is","it","as","an","by","at",
    "this","that","these","those","from","be","are","was","were","or","not","your", "job description"
}

# Create function here (spacy_keywords) to extract meaningul terms (lowercase+lemmatise, kepp nouns/proper nouns + 
# noun chunks, drop short /common words + digits + punctutation?)
def spacy_keywords(text: str):
    doc = nlp(text) # Tokenises the text to produce doc obj
    tokens = set()  # Empty set to store tokens

    # 1) Single word 'skill' tokens (nouns, proper nouns)
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.like_num:
            continue
        if tok.pos_ in {"NOUN","PROPN"}:
            lemma = tok.lemma_.lower().strip()  # Lemmatise, lowercase, strip 
            if lemma not in STOP_SHORT:
                tokens.add(lemma)
    
    # 2) Noun 'chunks'
    # Building the multi-word terms
    for chunk in doc.noun_chunks:
        chunk_text = " ".join(
            w.lemma_.lower() for w in chunk
            if not (w.is_stop or w.is_punct or w.like_num)   
        ).strip()
        if " " in chunk_text:
            tokens.add(chunk_text)
    
    return tokens

# Create a function here (score_match) to return the score, matched terms and missing terms. 
def score_match(cv_terms, jd_terms):
    """
        Weighted match:
        - Exact multi-word phrase match counts more
        - Single-word overlap counts less
    """

    # Add all the phrases + words in JD and CV into sets
    phrases_jd = {t for t in jd_terms if " " in t}
    words_jd = {t for t in jd_terms if " " not in t}

    phrases_cv = {t for t in cv_terms if " " in t}
    words_cv = {t for t in cv_terms if " " not in t}

    # Common phrases + words
    phrase_overlap = phrases_cv & phrases_jd
    word_overlap = words_cv & words_jd

    # Weights: phrases = 2, words = 1
    # Phrases have higher match score as more specific
    score_sum = 2*len(phrase_overlap) + len(word_overlap) 
    total = 2*len(phrases_jd) + len(words_jd)
    percent = 0.0 if total == 0 else round(100*score_sum / total, 1)

    matched = sorted(phrase_overlap | word_overlap)
    missing = sorted(list(jd_terms - cv_terms))

    return percent, matched, missing

# --- UI ---
cv_text = st.text_area("Paste your CV text")
jd_text = st.text_area("Paste the Job Description")

if st.button("Analyze"):
    if not cv_text or not jd_text:
        st.warning("Please paste both CV and JD.")
    else:
        cv_keywords = spacy_keywords(cv_text)
        jd_keywords = spacy_keywords(jd_text)

        pct, matched, missing = score_match(cv_keywords, jd_keywords)
        
        st.metric("Match %", f"{pct}%")
        # Two-column layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Matched terms")
            if matched:
                st.markdown("\n".join([f"- {t}" for t in matched[:80]]))
            else:
                st.write("(none)")

        with col2:
            st.subheader("❌ Missing terms")
            if missing:
                st.markdown("\n".join([f"- {t}" for t in missing[:80]]))
            else:
                st.write("(none)")

        # Basic suggestions
        if missing:
            st.subheader("💡 Suggestions")
            st.markdown(
                "Consider adding evidence for:\n" +
                "\n".join([f"- {t}" for t in missing[:10]])
            )
