# chatbot2.py

import os
import pickle
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load environment variables (especially for API key)
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b:free"

os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY is not set in your .env file.")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert career advisor and the world’s most accurate job matcher, capable of analyzing JSON data with precision and delivering deep, insightful evaluations.

Below is a JSON array of job postings (companies). When I give you a student profile (also in JSON), you must:

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
"""),
    ("user", """
🎓 Student Profile JSON:
{student}

(Use the above student JSON to evaluate all jobs thoroughly.) 
""")
])


def analyze_matches(pickle_file_path: str, student_data: list):
    """
    This function:
    - Loads job matches from a .pkl file.
    - Extracts relevant matches for the given student.
    - Sends them through a LangChain prompt for reasoning and analysis.

    Args:
        pickle_file_path: Path to the pickle file containing job matches.
        student_data: A list with one student dictionary.

    Returns:
        A string response with detailed match analysis.
    """
    # Load job matches from pickle
    with open(pickle_file_path, 'rb') as f:
        matches = pickle.load(f)

    # Extract student name
    student = student_data[0]
    student_name = f"{student.get('first_name', '')} {student.get('last_name', '')}".strip()

    # Get top job matches for the student
    top_jobs = matches.get(student_name, [])
    if not top_jobs:
        return f"❌ No jobs matched for student {student_name}."

    # Initialize LangChain components
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model_name=MODEL_NAME
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Prepare inputs
    response = chain.invoke({
        "companies": top_jobs,
        "student": student_data
    })

    return response
