import streamlit as st
import json
import pickle
from BM_25 import load_students, load_jsonl_file, preprocess_jobs, build_bm25_model, match_students_to_jobs
from chatbot3 import analyze_matches

# ────────────────────────── Streamlit UI ────────────────────────── #
st.title("🔍 Profeshare Job Matcher")

uploaded_file = st.file_uploader("📁 Upload student profile JSON file", type=["json"])
interest_input = st.text_input("💡 Enter interests (separated by '+')", placeholder="e.g. frontend+developer+intern")

if uploaded_file and interest_input:
    # Parse student JSON
    student_data = json.load(uploaded_file)
    if not isinstance(student_data, list):
        student_data = [student_data]

    # Update interests in student JSON
    interest_list = interest_input.split("+")
    for student in student_data:
        if "job_preferences" not in student:
            student["job_preferences"] = {}
        student["job_preferences"]["interests"] = interest_list

    # Save updated student.json temporarily
    with open("students.json", "w") as f:
        json.dump(student_data, f, indent=2)

    st.success("✅ Interests updated and student profile processed!")

    # ───── Load Jobs Data ───── #
    jobs = []
    for part_file in ["part_1.jsonl", "part_2.jsonl", "part_3.jsonl"]:
        data = load_jsonl_file(part_file)
        jobs.extend(data)

    # ───── Preprocess Jobs ───── #
    job_texts, job_index = preprocess_jobs(jobs)
    bm25 = build_bm25_model(job_texts)

    # ───── Match Students ───── #
    matches = match_students_to_jobs(student_data, jobs, bm25, job_index, top_n=10)

    # Save for chatbot3 input
    with open("student_job_matches.pkl", "wb") as file:
        pickle.dump(matches, file)

    st.success("🎯 Top job matches generated using BM25!")

    # ───── Reasoning with LLM (chatbot3) ───── #
    final_response = analyze_matches("student_job_matches.pkl", student_data)
    st.markdown("## 🤖 LLM Career Analysis")
    st.write(final_response)
