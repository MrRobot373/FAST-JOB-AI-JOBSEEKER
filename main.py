# import os
# import json
# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn
# import google.generativeai as genai
# from typing import Optional, Dict

# # -----------------------------------------------------------------------------
# # Gemini API Config
# # -----------------------------------------------------------------------------
# genai.configure(
#     api_key=os.environ.get(
#         "GEMINI_API_KEY",
#         "AIzaSyCPg-kwf9cAAjHkMCDVYY_t9yLNgC_StvM"
#     )
# )

# generation_config = {
#     "temperature": 1,
#     "top_p": 0.95,
#     "top_k": 40,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash-8b",
#     generation_config=generation_config,
# )

# # -----------------------------------------------------------------------------
# # Resume Analysis Function
# # -----------------------------------------------------------------------------
# def analyze_resume_with_gemini(resume_data):
#     resume_json_str = json.dumps(resume_data, indent=2)

#     prompt = f"""
#     You are a highly skilled AI Resume Analyst. Carefully analyze the provided resume 
#     (in JSON format below) and return **a structured, well-formatted response** like a professional resume consultant.

#     **🔍 Your Task:**
#     - Title the Resume** (e.g., "Confident", "Needs Improvement", "Underconfident").
#     - Rewrite the Summary Professionally** (Highlight skills and impact).
#     - Suggest Relevant Job Roles** (5-7 roles with a brief reason).
#     - List Additional Skills to Learn** (Practical, relevant, and industry-specific).
#     - Rate the Resume Level** (Beginner, Intermediate, Advanced + one-line explanation).
#     - Point Out Resume Improvements** (Structured sections with actionable feedback).
#     - Check for Grammar Mistakes** (Only mention if mistakes exist).

#     **📌 Candidate Resume Data (JSON)**:
#     {resume_json_str}

#     ** Format Your Response EXACTLY Like This** (keeping these headings):

#    Resume Title: <Your Title>

#      Professional Summary:
#     <Rewrite ONLY the summary from the resume's "otherDetails.summary" in a professional manner.>

#      Suggested Job Role:
#     1. <Role> - [Reason]
#     2. <Role> - [Reason]
#     ...
    
#      Additional Skills to Learn:
#     - <Skill> - [Why it is important]
#     - <Skill> - [Why it is important]
#     ...

#      Resume Rating: <Beginner/Intermediate/Advanced>
#     - <One-line explanation.>

#      Resume Improvement:
#     - Skills Section: "<Actionable improvement>"
#     - Experience Section: "<Actionable improvement>"
#     - Certifications Section: "<Actionable improvement>"
#     ...

#      Grammar Check:
#     - <Only mention if mistakes exist; otherwise, skip.>

#     **⚠️ Important:**  
#     - DO NOT generate made-up job roles or skills.  
#     - Base your analysis ONLY on the provided JSON data.  
#     - Keep responses clear, formatted, and actionable. 
#     - DO Not use bracetes in responses  
#     """
#     try:
#         chat_session = model.start_chat(history=[])
#         response = chat_session.send_message(prompt)
#         return response.text
#     except Exception as e:
#         return f"❌ API Request Error: {str(e)}"

# # -----------------------------------------------------------------------------
# # Q&A Generator (from question.py)
# # -----------------------------------------------------------------------------
# def generate_questions_with_answers(resume, jd=None):
#     jd_section = f"\nJob Description:\n{json.dumps(jd, indent=2)}" if jd else ""
#     source_instruction = (
#         "Blend Resume and JD to generate realistic and relevant questions."
#         if jd else
#         "Use only the Resume to generate questions."
#     )

#     prompt = f"""
# You are a senior interviewer preparing a mock interview for a candidate.

# 🎯 Based on the resume{' and job description' if jd else ''}, infer the most suitable role or domain.
# Then generate **20 interview questions with full, descriptive answers**:
# - For technical roles: prioritize domain-specific 70 % technical questions 30% Nontechnical questions.
# - For non-technical roles: include a mix of domain-relevant and behavioral questions.

# ✅ Format:
# - Use clear section headers as needed  (Technical, Non Technical) for technical roles only
# - Number each Q&A from **1 to 20** 
# - For each:
#   [number]. Q: [question]
#      A: [answer]

# - Ensure Q and A are on separate lines
# - Infer answers professionally even if information is limited
# {source_instruction}

# {jd_section}

# Resume:
# {json.dumps(resume, indent=2)}

# Now generate the full response.
# """
#     chat_session = model.start_chat(history=[])
#     response = chat_session.send_message(prompt)
#     return response.text.strip()

# # -----------------------------------------------------------------------------
# # JD Skill Expansion + Resume Enhancer (from generator.py)
# # -----------------------------------------------------------------------------
# def expand_jd_skills(jd_skills):
#     prompt = f"""
# Given the following job description (JD) skills, generate a JSON dictionary where each JD skill 
# is a key, and the value is a list of relevant or commonly associated skills.

# JD Skills: {json.dumps(list(jd_skills))}

# Return output as a valid JSON dictionary, nothing else.

# Example Output:
# {{
#     "React": ["Redux", "Angular", "Next.js", "Node.js"],
#     "Python": ["Django", "Flask", "FastAPI", "TensorFlow"]
# }}
# """
#     try:
#         response = model.chat(prompt)
#         return json.loads(response.text.strip())
#     except:
#         return {skill: [skill] for skill in jd_skills}

# def reorder_resume_by_skill_match(jd: dict, resume: dict) -> dict:
#     jd_skills = {s.lower().strip() for s in jd.get("skills", [])}
#     expanded_skill_map = expand_jd_skills(jd_skills)

#     expanded_jd_skills = set(jd_skills)
#     for related_skills in expanded_skill_map.values():
#         expanded_jd_skills.update(s.lower().strip() for s in related_skills)

#     def count_skill_matches(skill_list):
#         return sum(skill.lower() in expanded_jd_skills for skill in skill_list)

#     if "certificationDetails" in resume and isinstance(resume["certificationDetails"], list):
#         resume["certificationDetails"] = sorted(
#             resume["certificationDetails"],
#             key=lambda cert: count_skill_matches(cert.get("skills", [])),
#             reverse=True
#         )

#     if "professionalDetails" in resume and isinstance(resume["professionalDetails"], list):
#         resume["professionalDetails"] = sorted(
#             resume["professionalDetails"],
#             key=lambda prof: count_skill_matches(prof.get("skills", [])),
#             reverse=True
#         )

#     if "projectDetails" in resume and isinstance(resume["projectDetails"], list):
#         resume["projectDetails"] = sorted(
#             resume["projectDetails"],
#             key=lambda proj: count_skill_matches(proj.get("skills", [])),
#             reverse=True
#         )

#     if "otherDetails" in resume and isinstance(resume["otherDetails"], dict):
#         if "skills" in resume["otherDetails"] and isinstance(resume["otherDetails"]["skills"], list):
#             resume["otherDetails"]["skills"] = sorted(
#                 resume["otherDetails"]["skills"],
#                 key=lambda skill: skill.lower() in expanded_jd_skills,
#                 reverse=True
#             )

#     return resume

# def enhance_summary(jd: dict, resume: dict) -> dict:
#     if "otherDetails" in resume and isinstance(resume["otherDetails"], dict):
#         if "summary" in resume["otherDetails"] and isinstance(resume["otherDetails"]["summary"], str):
#             original_summary = resume["otherDetails"]["summary"]
#             new_summary = rewrite_summary(original_summary, jd)
#             resume["otherDetails"]["summary"] = new_summary
#     return resume

# def rewrite_summary(original_summary: str, jd: dict) -> str:
#     prompt = f"""
# Rewrite and enhance the following resume summary to better align with the job description (JD).
# - Keep all important details from the original summary.
# - Expand and integrate key job requirements from the JD.
# - Maintain professional, structured, and fluent writing.
# - Naturally include JD-specific skills as per JD and resume describing the responsibilities, and industry context.
# - Do NOT shorten,add or remove content, only enhance and improve relevance.

# ### Job Description:
# {json.dumps(jd, indent=2)}

# ### Original Resume Summary:
# {original_summary}

# ### Enhanced Resume Summary (with JD Integration):
# """
#     try:
#         response = model.generate_content(prompt)
#         new_summary = response.text.strip()

#         if len(new_summary) <= len(original_summary):
#             return f"{original_summary} (Additional JD-aligned refinements: {new_summary})"

#         return new_summary
#     except:
#         return original_summary

# # -----------------------------------------------------------------------------
# # FastAPI Setup
# # -----------------------------------------------------------------------------
# app = FastAPI(title="JOB_SEEKER")

# class ResumeData(BaseModel):
#     name: str = "User"
#     skills: list = []
#     experience: str = "Not provided"
#     education: str = "Not provided"
#     otherDetails: dict = {}

# class QAGeneratorRequest(BaseModel):
#     resume: dict
#     jd: Optional[Dict] = None

# class ResumeUpdaterRequest(BaseModel):
#     resume: dict
#     jd: dict

# # -----------------------------------------------------------------------------
# # Endpoints
# # -----------------------------------------------------------------------------
# @app.post("/ANALYSER")
# def analyze_resume_api(resume_data: ResumeData):
#     resume_dict = resume_data.dict()
#     analysis_result = analyze_resume_with_gemini(resume_dict)
#     return {"analysis": analysis_result}

# @app.post("/QUESTIONEER")
# def generate_qa_api(request: QAGeneratorRequest):
#     result = generate_questions_with_answers(request.resume, request.jd)
#     return {"qa_output": result}

# @app.post("/RESUME_GENERATOR")
# def resume_updater(request: ResumeUpdaterRequest):
#     reordered = reorder_resume_by_skill_match(request.jd, request.resume)
#     enhanced = enhance_summary(request.jd, reordered)
#     return {"updated_resume": enhanced}

# # -----------------------------------------------------------------------------
# # Uvicorn entry point
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)

import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
from typing import Optional, Dict

# -----------------------------------------------------------------------------
# Gemini API Config
# -----------------------------------------------------------------------------
genai.configure(
    api_key=os.environ.get(
        "GEMINI_API_KEY",
        "AIzaSyAI7lzHO-MfOEMYM0ttQdY230wTVMx_4Rs"
    )
)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite",
    generation_config=generation_config,
)

# -----------------------------------------------------------------------------
# Resume Analysis Function
# -----------------------------------------------------------------------------
def analyze_resume_with_gemini(resume_data):
    resume_json_str = json.dumps(resume_data, indent=2)

    prompt = f"""
    You are a highly skilled AI Resume Analyst. Carefully analyze the provided resume
    (in JSON format below) and return **a structured, well-formatted response** like a professional resume consultant.

    **🔍 Your Task:**
    - Title the Resume** (e.g., "Confident", "Needs Improvement", "Underconfident").
    - Rewrite the Summary Professionally** (Highlight skills and impact). Ensure it's concise (around 70 words).
    - Suggest Relevant Job Roles** (5-7 roles with a brief reason).
    - List Additional Skills to Learn** (Practical, relevant, and industry-specific).
    - Rate the Resume Level** (Beginner, Intermediate, Advanced + one-line explanation).
    - Point Out Resume Improvements** (Structured sections with actionable feedback).
    - Check for Grammar Mistakes** (Only mention if mistakes exist).

    **📌 Candidate Resume Data (JSON)**:
    {resume_json_str}

    ** Format Your Response EXACTLY Like This** (keeping these headings):

   Resume Title: <Your Title>

     Professional Summary:
    <Rewrite ONLY the summary from the resume's "otherDetails.summary" in a professional manner, and keep it short (around 70 words).>

     Suggested Job Role:
    1. <Role> - [Reason]
    2. <Role> - [Reason]
    ...
   
     Additional Skills to Learn:
    - <Skill> - [Why it is important]
    - <Skill> - [Why it is important]
    ...

     Resume Rating: <Beginner/Intermediate/Advanced>
    - <One-line explanation.>

     Resume Improvement:
    - Skills Section: "<Actionable improvement>"
    - Experience Section: "<Actionable improvement>"
    - Certifications Section: "<Actionable improvement>"
    ...

     Grammar Check:
    - <Only mention if mistakes exist; otherwise, skip.>

    **⚠️ Important:**  
    - DO NOT generate made-up job roles or skills.  
    - Base your analysis ONLY on the provided JSON data.  
    - Keep responses clear, formatted, and actionable.
    - DO Not use bracetes in responses  

    """
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        return f"❌ API Request Error: {str(e)}"
   
def clean_response(response: str):
    """
    Post-processes the AI response to remove arrays and format the content.
    This ensures that skills, job roles, etc., are in human-readable formats.
    """
    # Handle array-like structures (e.g., ["Java", "C++"] becomes "Java, C++")
    response = response.replace("[", "").replace("]", "").replace(",", "")
    response = response.replace("\n", " ").strip()  # Ensure no extra newlines
    return response


# -----------------------------------------------------------------------------
# Q&A Generator (from question.py)
# -----------------------------------------------------------------------------
def generate_questions_with_answers(resume, jd=None):
    jd_section = f"\nJob Description:\n{json.dumps(jd, indent=2)}" if jd else ""
    source_instruction = (
        "Blend Resume and JD to generate realistic and relevant questions."
        if jd else
        "Use only the Resume to generate questions."
    )

    prompt = f"""
You are a senior interviewer preparing a mock interview for a candidate.

🎯 Based on the resume{' and job description' if jd else ''}, infer the most suitable role or domain.
Then generate **20 interview questions with full, descriptive answers**:
- For technical roles: prioritize domain-specific 70 % technical questions 30% Nontechnical questions.
- For non-technical roles: include a mix of domain-relevant and behavioral questions.

✅ Format:
- Use clear section headers as needed  (Technical, Non Technical) for technical roles only
- Number each Q&A from **1 to 20**
- For each:
  [number]. Q: [question]
     A: [answer]

- Ensure Q and A are on separate lines
- Infer answers professionally even if information is limited
{source_instruction}

{jd_section}

Resume:
{json.dumps(resume, indent=2)}

Now generate the full response.
"""
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text.strip()

# -----------------------------------------------------------------------------
# JD Skill Expansion + Resume Enhancer (from generator.py)
# -----------------------------------------------------------------------------
def expand_jd_skills(jd_skills):
    prompt = f"""
Given the following job description (JD) skills, generate a JSON dictionary where each JD skill
is a key, and the value is a list of relevant or commonly associated skills.

JD Skills: {json.dumps(list(jd_skills))}

Return output as a valid JSON dictionary, nothing else.

Example Output:
{{
    "React": ["Redux", "Angular", "Next.js", "Node.js"],
    "Python": ["Django", "Flask", "FastAPI", "TensorFlow"]
}}
"""
    try:
        response = model.chat(prompt)
        return json.loads(response.text.strip())
    except:
        return {skill: [skill] for skill in jd_skills}

def reorder_resume_by_skill_match(jd: dict, resume: dict) -> dict:
    jd_skills = {s.lower().strip() for s in jd.get("skills", [])}
    expanded_skill_map = expand_jd_skills(jd_skills)

    expanded_jd_skills = set(jd_skills)
    for related_skills in expanded_skill_map.values():
        expanded_jd_skills.update(s.lower().strip() for s in related_skills)

    def count_skill_matches(skill_list):
        return sum(skill.lower() in expanded_jd_skills for skill in skill_list)

    if "certificationDetails" in resume and isinstance(resume["certificationDetails"], list):
        resume["certificationDetails"] = sorted(
            resume["certificationDetails"],
            key=lambda cert: count_skill_matches(cert.get("skills", [])),
            reverse=True
        )

    if "professionalDetails" in resume and isinstance(resume["professionalDetails"], list):
        resume["professionalDetails"] = sorted(
            resume["professionalDetails"],
            key=lambda prof: count_skill_matches(prof.get("skills", [])),
            reverse=True
        )

    if "projectDetails" in resume and isinstance(resume["projectDetails"], list):
        resume["projectDetails"] = sorted(
            resume["projectDetails"],
            key=lambda proj: count_skill_matches(proj.get("skills", [])),
            reverse=True
        )

    if "otherDetails" in resume and isinstance(resume["otherDetails"], dict):
        if "skills" in resume["otherDetails"] and isinstance(resume["otherDetails"]["skills"], list):
            resume["otherDetails"]["skills"] = sorted(
                resume["otherDetails"]["skills"],
                key=lambda skill: skill.lower() in expanded_jd_skills,
                reverse=True
            )

    return resume

def enhance_summary(jd: dict, resume: dict) -> dict:
    if "otherDetails" in resume and isinstance(resume["otherDetails"], dict):
        if "summary" in resume["otherDetails"] and isinstance(resume["otherDetails"]["summary"], str):
            original_summary = resume["otherDetails"]["summary"]
            new_summary = rewrite_summary(original_summary, jd)
            resume["otherDetails"]["summary"] = new_summary
    return resume

def rewrite_summary(original_summary: str, jd: dict) -> str:
    prompt = f"""
Rewrite and enhance the following resume summary to better align with the job description (JD).
- Keep all important details from the original summary.
- Expand and integrate key job requirements from the JD.
- Maintain professional, structured, and fluent writing.
- Naturally include JD-specific skills as per JD and resume describing the responsibilities, and industry context.
- Keep it short (around 70 words), and ensure the summary does not mention skills as an array (i.e., no brackets).
- Do NOT shorten, add, or remove content, only enhance and improve relevance.

### Job Description:
{json.dumps(jd, indent=2)}

### Original Resume Summary:
{original_summary}

### Enhanced Resume Summary (with JD Integration):
"""
    try:
        response = model.generate_content(prompt)
        new_summary = response.text.strip()

        # Ensure the summary is around 70 words
        if len(new_summary.split()) > 70:
            return ' '.join(new_summary.split()[:70])  # Trim to 70 words max

        return new_summary
    except:
        return original_summary

# -----------------------------------------------------------------------------
# FastAPI Setup
# -----------------------------------------------------------------------------
app = FastAPI(title="JOB_SEEKER")

class ResumeData(BaseModel):
    name: str = "User"
    skills: list = []
    experience: str = "Not provided"
    education: str = "Not provided"
    otherDetails: dict = {}

class QAGeneratorRequest(BaseModel):
    resume: dict
    jd: Optional[Dict] = None

class ResumeUpdaterRequest(BaseModel):
    resume: dict
    jd: dict

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/ANALYSER")
def analyze_resume_api(resume_data: ResumeData):
    resume_dict = resume_data.dict()
    analysis_result = analyze_resume_with_gemini(resume_dict)
    return {"analysis": analysis_result}

@app.post("/QUESTIONEER")
def generate_qa_api(request: QAGeneratorRequest):
    result = generate_questions_with_answers(request.resume, request.jd)
    return {"qa_output": result}

@app.post("/RESUME_GENERATOR")
def resume_updater(request: ResumeUpdaterRequest):
    reordered = reorder_resume_by_skill_match(request.jd, request.resume)
    enhanced = enhance_summary(request.jd, reordered)
    return {"updated_resume": enhanced}

# -----------------------------------------------------------------------------
# Uvicorn entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)
