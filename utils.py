"""
Core logic for the Resume/JD Matcher.
Keeps text extraction, cleaning, and scoring separate from the Streamlit UI
so it's easy to test and easy to reuse (e.g. behind an API later).
"""

import re
from io import BytesIO

import nltk
from pypdf import PdfReader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download once; Streamlit Cloud will run this on first boot and cache it.
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

# Common resume filler words that aren't useful "skills" for matching
GENERIC_WORDS = {
    "experience", "work", "team", "project", "projects", "responsible",
    "ability", "strong", "using", "used", "including", "years", "year",
    "role", "company", "job", "candidate", "please", "will", "must",
    "looking", "plus", "good", "great", "skills", "skill", "knowledge",
    "familiar", "familiarity", "requirements", "required", "preferred",
}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using pypdf."""
    reader = PdfReader(BytesIO(file_bytes))

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def clean_text(raw_text: str) -> str:
    """Lowercase, strip punctuation/numbers, collapse whitespace."""
    text = raw_text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_keywords(text: str, top_n: int = 30) -> list[str]:
    """
    Extract the most relevant single-word and two-word phrases from text
    using TF-IDF, filtering out stopwords and generic resume filler.
    """
    cleaned = clean_text(text)
    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 2),
        max_features=200,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([cleaned])
    except ValueError:
        # Text too short / empty after cleaning
        return []

    scores = tfidf_matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)

    keywords = []
    for term, score in ranked:
        if score <= 0:
            continue
        words_in_term = term.split()
        if any(w in GENERIC_WORDS for w in words_in_term):
            continue
        if any(len(w) <= 2 for w in words_in_term):
            continue
        keywords.append(term)
        if len(keywords) >= top_n:
            break

    return keywords


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Return a 0-100 similarity score between resume and job description
    using TF-IDF vectorization + cosine similarity.
    """
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    if not resume_clean or not jd_clean:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    except ValueError:
        return 0.0

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 1)


def find_missing_keywords(resume_text: str, jd_text: str, top_n: int = 20) -> list[str]:
    """
    Return keywords that appear in the JD's top terms but are absent
    (or only weakly present) in the resume.
    """
    jd_keywords = get_keywords(jd_text, top_n=top_n)
    resume_clean = clean_text(resume_text)

    missing = [kw for kw in jd_keywords if kw not in resume_clean]
    return missing


def find_matching_keywords(resume_text: str, jd_text: str, top_n: int = 20) -> list[str]:
    """Return JD top keywords that DO appear in the resume."""
    jd_keywords = get_keywords(jd_text, top_n=top_n)
    resume_clean = clean_text(resume_text)

    matching = [kw for kw in jd_keywords if kw in resume_clean]
    return matching
