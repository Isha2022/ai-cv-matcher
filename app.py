import streamlit as st
import string
import spacy
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import Dict, Set, Tuple, List


# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

st.set_page_config(page_title="CV ↔ JD Matcher", layout="wide")
st.title("CV ↔ Job Description Matcher (MVP)")
st.caption("Paste your CV and a Job Description. The app extracts skills/tools and shows the overlap.")

# Function to extract skills/tools/domain/seniority, default source to "CV"
def extract_skills_ai(text: str, source: str = "cv"):
   
    # Guidance based on source
    if source == "cv":
        guidance = (
            "Extract only capabilities explicitly evidenced in the CV. "
            "Do not infer beyond evidence."
        )
    else:
        guidance = ("Extract required and preferred capabilities from the job description. "
            "Infer likely seniority as one of: intern, junior, mid-level, senior."
        )
    
    # System message - return the info JSON
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant that extracts capabilities from text and returns JSON only. "
            "Rules: lowercase all terms; no duplicates; prefer canonical names "
            "(e.g., 'postgresql', 'tensorflow', 'rest api'). "
            "If info is missing, return an empty array. Return strictly valid JSON in this shape:\n"
            "{"
            '"skills": [],'
            '"tools": [],'
            '"domains": [],'
            '"seniority": "intern|junior|mid-level|senior|unknown",'
            '"suggestions": []'
            "}"
            "Definitions: "
            "- skills: human capabilities (e.g., accessibility, responsive design, performance optimisation)."
            "- tools: named technologies/frameworks (e.g., react, next.js, typescript, figma, mui, mantine, cypress)."
            "- domains: business/industry/problem areas (e.g., frontend web development, it service management, service desk operations, e-commerce, fintech, healthcare, data platforms)."
        ),
    }
    
    # User message w/ cv or jd
    user_msg = {
        "role": "user",
        "content": f"{guidance}\n\nTEXT:\n{text}",
    }

    # Make API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_msg, user_msg],
        temperature=0,  # deterministic -> Same input, same output
    )

    # Parse JSON safely
    raw_output = response.choices[0].message.content.strip()    # Get AI reply as text
    try:                                                        # Try to convert JSON string into dictionary
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(raw_output[start:end+1])
            except Exception:
                data = {}
        else:
            data = {}

    # Ensure keys exist
    for k, v in {
        "skills": [],
        "tools": [],
        "domains": [],
        "seniority": "unknown",
        "suggestions": [],
    }.items():
        data.setdefault(k, v)

    # Normalise to lowercase, strip, dedupe
    def norm_list(lst):
        if not isinstance(lst, list):   # Make sure value is a list
            return []
        out = []
        seen = set()
        for x in lst:
            if not isinstance(x, str):  # Only keep strings
                continue
            t = x.strip().lower()       # Trim spaces + convert to lowercase
            if t and t not in seen:     # Remove duplicates
                seen.add(t)
                out.append(t)
        return out

    # Clean all lists
    data["skills"] = norm_list(data["skills"])
    data["tools"] = norm_list(data["tools"])
    data["domains"] = norm_list(data["domains"])
    data["suggestions"] = norm_list(data["suggestions"])
    if not isinstance(data["seniority"], str):  # Default to unknown
        data["seniority"] = "unknown"
    else:
        data["seniority"] = data["seniority"].strip().lower()

    return data

# Known tooling/tech terms we prefer to keep under "tools"
KNOWN_TOOLS = {
    # frameworks & libs
    "django","flask","fastapi","spring","react","nodejs","express","next.js","pytorch","tensorflow","keras",
    "sklearn","scikit-learn","pandas","numpy","matplotlib","seaborn","spaCy","spacy","nltk","opencv",
    # cloud & devops
    "aws","azure","gcp","google cloud","docker","kubernetes","terraform","github actions","jenkins","gitlab ci",
    # data/infra
    "postgres","postgresql","mysql","sqlite","mongodb","redis","kafka","spark","hadoop","snowflake","bigquery",
    # web/api/testing
    "rest","rest api","graphql","grpc","pytest","jest","playwright","cypress",
}
ALIASES = {
    "node": "nodejs",
    "node.js": "nodejs",
    "postgreSQL": "postgresql",
    "postgre": "postgresql",
    "js": "javascript",
    "ts": "typescript",
    "google cloud platform": "google cloud",
    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "tf": "tensorflow",
    "ml": "machine learning",
    "nlp": "natural language processing",
}

# Lowercase + Trim + if in aliases then return mapped canonical vers.
def canonicalise(term: str) -> str:
    t = term.lower().strip()
    return ALIASES.get(t, t)

# Move known tools into tools bucket, keep others as skills
def rebucket(data: Dict) -> Dict:
    skills = {canonicalise(t) for t in data.get("skills", [])}  # Get value under 'skills' key, else []
    tools = {canonicalise(t) for t in data.get("tools", [])}
    domains = {canonicalise(t) for t in data.get("domains", [])}

    # If term is in skills and is a 'known tool' move to tool
    skills_in_tools = {s for s in skills if s in KNOWN_TOOLS}
    skills = skills - skills_in_tools
    tools = tools | skills_in_tools
    
    overlap = skills & tools
    skills = skills - overlap

    sd_overlap = skills & domains
    domains = domains - sd_overlap

    data["skills"] = sorted(skills)
    data["tools"]  = sorted(tools)
    data["domains"]  = sorted(domains)
    return data

# Matching + Sorting
def compute_match(cv: Dict, jd: Dict) -> Tuple[float, List[str], List[str]]:
    def as_set(d: Dict) -> Set[str]:
        # union of skills + tools + domains
        return set(d.get("skills", [])) | set(d.get("tools", [])) | set(d.get("domains", []))

    def calc_weighting(text: str) -> int:
        return 2 if " " in text else 1
    
    cv_all = as_set(cv)
    jd_all = as_set(jd)
    
    matched = sorted(cv_all & jd_all)
    missing = sorted(jd_all - cv_all)

    total_weight = sum(calc_weighting(t) for t in jd_all)
    matched_weight = sum(calc_weighting(t) for t in matched)
    if total_weight == 0:
        return 0.0, matched, missing
    pct = round((matched_weight / total_weight) * 100, 1)

    return pct, matched, missing


# --- UI ---
col1, col2 = st.columns(2)
with col1:
    cv_text = st.text_area("Paste your CV text", height=260, placeholder="Paste the *text* of your CV here.")
with col2:
    jd_text = st.text_area("Paste the Job Description", height=260, placeholder="Paste the job posting text here.")

analyse = st.button("Analyse", type="primary")

if analyse:
    if not cv_text.strip() or not jd_text.strip():
        st.warning("Please paste both CV and Job Description text.")
        st.stop()

    with st.spinner("Extracting capabilities with AI..."):
        cv_raw = extract_skills_ai(cv_text, source="cv")
        jd_raw = extract_skills_ai(jd_text, source="jd")

    # Rebucket to keep tools consistent (e.g., tensorflow → tools)
    cv_data = rebucket(cv_raw)
    jd_data = rebucket(jd_raw)

    # Compute match
    pct, matched, missing = compute_match(cv_data, jd_data)

    st.metric("Match %", f"{pct}%")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("CV (normalised):")
        st.badge("**Skills:** ")
        st.write("\n".join([f"- {s}" for s in cv_data["skills"]]) or "(none)")
        st.badge("**Tools:** ", color="red")
        st.write("\n".join([f"- {s}" for s in cv_data["tools"]]) or "(none)")        
        st.badge("**Domains:** ", color="violet")
        st.write("\n".join([f"- {s}" for s in cv_data["domains"]]) or "(none)")

    with c2:
        st.subheader("JD (normalised):")
        st.badge("**Skills** ")
        st.write("\n".join([f"- {s}" for s in jd_data["skills"]]) or "(none)")
        st.badge("**Tools:** ", color="red")
        st.write("\n".join([f"- {s}" for s in jd_data["tools"]]) or "(none)")        
        st.badge("**Domains:** ", color="violet")
        st.write("\n".join([f"- {s}" for s in jd_data["domains"]]) or "(none)")
        st.caption(f"Seniority (JD): **{jd_data.get('seniority','unknown')}**")
    
    with c3:
        st.subheader("Suggestions:")
        sugg = jd_data.get("suggestions", [])
        if missing and len(sugg) < 3:
            # Add deterministic suggestions from top missing items
            sugg = list(sugg) + list(missing[:5])
        if sugg:
            st.markdown("Add concrete points demonstrating: ")
            st.markdown("\n".join(f"- {s}" for s in sugg[:8]))
        else:
            st.write("(none)")

    st.divider()
    c4, c5= st.columns(2)
    with c4:
        st.subheader("✅ Matched")
        st.write("\n".join([f"- {m}" for m in matched]) if matched else "(none)")

    with c5:
        st.subheader("❌ Missing (from your CV)")
        st.write("\n".join([f"- {m}" for m in missing]) if missing else "(none)")

