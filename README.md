# 📄 Resume ↔ Job Description Matcher

A Streamlit web app that compares a resume against a job description and returns:

live demo :https://resume-analyzer-bbjpfc25nkkjtpsjjm924y.streamlit.app/
- A **match score** (0–100%) using TF-IDF + cosine similarity
- **Keywords you already have** that match the job description
- **Keywords missing** from your resume that the job description emphasizes

## Why this project

Applicant Tracking Systems (ATS) used by most companies do a version of this
keyword-matching before a human ever sees a resume. This project demonstrates:

- NLP fundamentals (TF-IDF vectorization, stopword removal, cosine similarity)
- Turning a data/ML pipeline into a usable, interactive tool (not just a notebook)
- Clean separation of logic (`utils.py`) from UI (`app.py`) — testable and reusable
- End-to-end shipping: local dev → tested → deployed with a public link

## Tech stack

- **Python 3.10+**
- **Streamlit** — web UI
- **scikit-learn** — TF-IDF vectorization, cosine similarity
- **NLTK** — stopword filtering
- **pdfplumber** — PDF text extraction

## Run locally

```bash
git clone <your-repo-url>
cd resume-analyzer
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy for free (Streamlit Community Cloud)

1. Push this project to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repo, branch (`main`), and set the main file
   to `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically and
   gives you a public URL like `https://your-app-name.streamlit.app`.

That live link is what goes on your resume/LinkedIn — not just the repo link.

## Project structure

```
resume-analyzer/
├── app.py           # Streamlit UI
├── utils.py         # Core matching logic (text extraction, TF-IDF, scoring)
├── requirements.txt # Dependencies
└── README.md
```

## Possible future enhancements

- Support `.docx` resumes in addition to PDF
- Use spaCy for smarter phrase/entity extraction instead of raw TF-IDF n-grams
- Add a "suggested resume bullet" generator using an LLM for missing skills
- Deploy the core logic as a FastAPI endpoint so it can be reused outside the UI
