"""
Resume / Job Description Matcher
---------------------------------
Upload (or paste) a resume and a job description. Get:
  - A match score (TF-IDF cosine similarity)
  - Keywords from the JD that ARE in your resume
  - Keywords from the JD that are MISSING from your resume

Run locally:    streamlit run app.py
Deploy free:    https://share.streamlit.io  (Streamlit Community Cloud)
"""

import streamlit as st

from utils import (
    compute_similarity,
    extract_text_from_pdf,
    find_matching_keywords,
    find_missing_keywords,
)

st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="centered")

st.title("📄 Resume ↔ Job Description Matcher")
st.write(
    "Paste a job description and upload (or paste) your resume. "
    "Get a match score and see exactly which keywords you're missing."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    resume_file = st.file_uploader("Upload PDF", type=["pdf"], key="resume_pdf")
    resume_text_input = st.text_area(
        "...or paste resume text",
        height=220,
        placeholder="Paste your resume text here if you'd rather not upload a PDF.",
    )

with col2:
    st.subheader("Job Description")
    jd_text_input = st.text_area(
        "Paste the job description",
        height=280,
        placeholder="Paste the full job description text here.",
    )

analyze_clicked = st.button("Analyze Match", type="primary", use_container_width=True)

if analyze_clicked:
    # Resolve resume text: prefer uploaded PDF, fall back to pasted text
    resume_text = ""
    if resume_file is not None:
        resume_text = extract_text_from_pdf(resume_file.read())
    elif resume_text_input.strip():
        resume_text = resume_text_input

    jd_text = jd_text_input

    if not resume_text.strip():
        st.error("Please upload a resume PDF or paste your resume text.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analyzing..."):
            score = compute_similarity(resume_text, jd_text)
            matched = find_matching_keywords(resume_text, jd_text)
            missing = find_missing_keywords(resume_text, jd_text)

        st.divider()
        st.subheader("Results")

        # Score with simple color-coded feedback
        if score >= 60:
            st.success(f"Match Score: {score}%  — Strong match")
        elif score >= 35:
            st.warning(f"Match Score: {score}%  — Moderate match, some gaps")
        else:
            st.error(f"Match Score: {score}%  — Low match, resume needs work")

        st.progress(min(int(score), 100) / 100)

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("**✅ Keywords you already have**")
            if matched:
                st.write(", ".join(matched))
            else:
                st.write("None detected — try adding more JD-specific terms.")

        with res_col2:
            st.markdown("**⚠️ Keywords missing from your resume**")
            if missing:
                st.write(", ".join(missing))
                st.caption(
                    "Consider naturally weaving these into your resume "
                    "(only if you genuinely have that experience!)."
                )
            else:
                st.write("Great — no major gaps detected.")

st.divider()
st.caption(
    "Built with Streamlit, scikit-learn (TF-IDF + cosine similarity), and NLTK. "
    "Runs entirely in-session — no data is stored."
)
