# CV ↔ Job Description Matcher

An AI-powered web application that compares the given CV against a job description, highlights matched and missing skills/tools, and provides suggestions for improvement + **a match % score**.

🚀 [[Try it live on Streamlit Cloud]](https://isha2022-ai-cv-matcher-app-robbvj.streamlit.app/)

---
## 🛠️ Tech Stack
- **Python 3.10+**
- [Streamlit](https://streamlit.io/) – front-end UI
- [OpenAI GPT-3.5](https://platform.openai.com/) – skill extraction
- [spaCy](https://spacy.io/) – NLP basic version ? 

---
## How To Run

1. **Clone this repo**
  And navigate to the directory
2. **Create virtual environment & Install dependencies**
```
  python -m venv .venv
  source .venv/bin/activate   # or .venv\Scripts\activate on Windows
  pip install -r requirements.txt
```
3. **Add your OpenAI API key**
  Create a .env file:
```
  OPENAI_API_KEY=sk-xxxx...
```
4. **Run the app**
```
  streamlit run app.py
```


