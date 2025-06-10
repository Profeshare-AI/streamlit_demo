# app.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
import pickle
import json 

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#getting data from BM-25
jobs = []
with open('student_job_matches.pkl', 'rb') as file:
    matches = pickle.load(file)

for student, job in matches.items():
    # If job is a JSON string, parse it
    if isinstance(job, str):
        job_dict = json.loads(job)
    else:
        job_dict = job  # already a dict

    jobs.append(job_dict)


# Ensure your OPENROUTER_API_KEY is set in your .env file
# Example in .env: OPENROUTER_API_KEY="sk-or-your-actual-openrouter-key"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop the Streamlit app if the key is missing

st.title("Profeshare Demo With Deepseek (via OpenRouter)")

# ─── 1. Paste your array of job-postings JSON here ────────────────────────────────
with open('students.json', 'r') as f:
    students = json.load(f)
COMPANIES_JSON = jobs
STUDENT_JSON = students
 # ───────────────────────────────────────────────────────────────────────────────


# ─── 2. Build a ChatPromptTemplate that embeds COMPANIES_JSON ────────────────────
prompt = ChatPromptTemplate.from_messages(
[
        (
            "system",
            """
You are an expert career advisor and the world’s most accurate job matcher, capable of analyzing JSON data with precision and delivering deep, insightful evaluations.

Below is a JSON array of job postings companies. When I give you a student profile (also in JSON), you must:

🔍 1. **Analyze each job posting minutely** for:
   - Required and preferred skills
   - Work type (internship/full-time)
   - Start date, job title, location preferences
   - Domain fit (e.g., software, AI, management, etc.)
   - Any additional qualification criteria

🎯 2. **Assign a “Match Score” (0–100%)** to each job based on how well the student's profile aligns.

📊 3. **Sort all jobs in ascending order** of Match Score (least to most relevant).

💡 4. For each job, provide a detailed but reader-friendly breakdown:

🧾 🧾 **Job X (Start Date: YYYY-MM-DD)** – “{{Job Title}} at {{Company Name}}”
- 🔢 **Match Score**: XX%
- ✅ **Why it's a good fit**:
  • ...
  • ...
- ⚠️ **Potential difficulties / mismatches**:
  • ...
  • ...

🎨 5. **Format the output beautifully** using bullet points, emojis, and bold headers. The goal is to make the analysis engaging, thorough, and useful for both career coaches and students.

⚠️ Do NOT output anything in JSON. Only return a clean, structured, and human-readable evaluation.

Here are the company profiles (job postings):

{companies}

(End of job postings.)
"""
        ),
        (
            "user",
            """
🎓 Student Profile JSON:
{student}

(Use the above student JSON to evaluate all jobs thoroughly.) 
"""
        ),
    ]
)


# ─── 3. Initialize your LLM & chain using OpenRouter ───────────────────────────────────
# Define the model name for Deepseek R1 0528 Qwen3 8B (free)
OPENROUTER_MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b:free"

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY, # Use api_key for the OpenRouter API key
    model_name=OPENROUTER_MODEL_NAME # Adjust max_tokens as needed for detailed responses
    # Optional: Add extra headers for OpenRouter leaderboards
    # openai_extra_headers={
    #     "HTTP-Referer": "https://your-profeshare-app.com", # Replace with your app's URL
    #     "X-Title": "Profeshare Demo", # Replace with your app's name
    # }
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


# ─── 4. Streamlit input box treats whatever the user types as {student} ──────────
input_text = st.text_input("Paste the student-profile JSON here (including all fields)")

if input_text:
    # Invoke with a dict containing our only variable, “student”
    # Ensure to use the COMPANIES_JSON and input_text (for student)
    answer = chain.invoke({"companies": COMPANIES_JSON, "student": STUDENT_JSON})
    st.write(answer)

