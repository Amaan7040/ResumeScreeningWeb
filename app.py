import os
import pickle
import re
import docx
import pdfplumber
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import mysql.connector

# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Define common skills
common_skills = {
    skill.lower()
    for skill in {
        "Python", "Java", "C++", "C", "JavaScript", "HTML", "CSS",
        "TypeScript", "Swift", "Kotlin", "Go", "Ruby", "PHP", "R", "MATLAB",
        "Perl", "Rust", "Dart", "Scala", "Shell Scripting", "React", "Angular",
        "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js",
        "Laravel", "Bootstrap", "TensorFlow", "PyTorch", "Keras",
        "Scikit-learn", "NLTK", "Pandas", "NumPy", "SQL", "MySQL",
        "PostgreSQL", "MongoDB", "Firebase", "Cassandra", "Oracle", "Redis",
        "MariaDB", "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
        "Terraform", "CI/CD", "Jenkins", "Git", "GitHub", "Cybersecurity",
        "Penetration Testing", "Ubuntu", "Ethical Hacking", "Firewalls",
        "Cryptography", "IDS", "Network Security", "Machine Learning",
        "Deep Learning", "Numpy", "Pandas", "Matplotlib", "Computer Vision",
        "NLP", "Big Data", "Hadoop", "Spark", "Data Analytics", "Power BI",
        "Tableau", "Data Visualization", "Reinforcement Learning",
        "Advanced DSA", "DSA", "Data Structures and Algorithm", "DevOps", "ML",
        "DL", "Image Processing", "JIRA", "Postman", "Excel", "Leadership",
        "Problem-Solving", "Communication", "Time Management", "Adaptability",
        "Teamwork", "Presentation Skills", "Critical Thinking",
        "Decision Making", "Public Speaking", "Project Management"
    }
}

abbreviation_map = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ds": "data science",
    "js": "javascript",
    "html": "hypertext markup language",
    "css": "cascading style sheets",
    "sql": "structured query language",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "dsa": "data structure algorithm"
}

# ---------------- Database Connections ----------------
def get_db_connection(db_name):
    """Establishes and returns a connection to the specified MySQL database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="amaan@khan704093",
        database=db_name,
        auth_plugin="mysql_native_password"
    )

# ---------------- Resume Processing Functions ----------------
def extract_text_from_file(file):
    text = ""
    if file.filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"[ERROR] PDF Processing Failed: {e}")
            return None
    elif file.filename.endswith(".docx"):
        try:
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"[ERROR] DOCX Processing Failed: {e}")
            return None
    return text.strip() if text.strip() else None

def extract_skills(text):
    extracted_skills = set()
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in common_skills:
            extracted_skills.add(token.text.lower())
    return list(extracted_skills)

def extract_name(text):
    lines = text.split('\n')
    return lines[0].strip() if lines else None

def load_model_and_vectorizer():
    try:
        with open("model.pkl", "rb") as model_file:
            rf = pickle.load(model_file)
        with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        return rf, tfidf
    except Exception as e:
        print(f"[ERROR] Failed to load model/vectorizer: {e}")
        return None, None

def process_resume(file):
    rf, tfidf = load_model_and_vectorizer()
    if not rf or not tfidf:
        return "[ERROR] ML model is missing!", None, None, None
    
    text = extract_text_from_file(file)
    if not text:
        return "[ERROR] Invalid or unsupported file format!", None, None, None
    
    user_name = extract_name(text)
    extracted_skills = extract_skills(text)
    
    try:
        text_vectorized = tfidf.transform([text])
        predicted_job = rf.predict(text_vectorized)[0]
        return None, predicted_job, extracted_skills, user_name
    except Exception as e:
        return f"[ERROR] Prediction failed: {e}", None, extracted_skills, user_name

def normalize_skill(skill):
    """Normalize a skill by converting abbreviations to full forms."""
    skill_lower = skill.lower()  # Make the skill lowercase to handle case sensitivity
    return abbreviation_map.get(skill_lower, skill_lower)  # Return full form if abbreviation exists, otherwise return the original skill

def compare_skills(predicted_job, extracted_skills, user_name):
    try:
        # Connect to skills database
        conn_skills = get_db_connection("skills_db")
        cursor_skills = conn_skills.cursor(dictionary=True)
        cursor_skills.execute("SELECT skills FROM JobRolesSkills WHERE job_role = %s", (predicted_job,))
        job_data = cursor_skills.fetchone()
        cursor_skills.close()
        conn_skills.close()
        
        if not job_data:
            print("[ERROR] Job role not found in database.")
            return []
        
        # Normalize the required skills from the database
        required_skills = set(normalize_skill(skill) for skill in job_data["skills"].split(", "))
        
        # Normalize the extracted skills
        normalized_extracted_skills = set(normalize_skill(skill) for skill in extracted_skills)
        
        missing_skills = required_skills - normalized_extracted_skills

        if missing_skills:
            # Connect to mismatch database
            conn_mismatch = get_db_connection("recommended_skills_db")
            cursor_mismatch = conn_mismatch.cursor()
            cursor_mismatch.execute(
                "INSERT INTO recommendskills (name, job_role, missing_skills) VALUES (%s, %s, %s)",
                (user_name, predicted_job, ", ".join(missing_skills))
            )
            conn_mismatch.commit()
            cursor_mismatch.close()
            conn_mismatch.close()
        
        return list(missing_skills)
    except Exception as e:
        print(f"[ERROR] Skill comparison failed: {e}")
        return []

# ---------------- Flask Application ----------------
app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_job = None
    error_message = None
    extracted_skills = []
    missing_skills = []
    user_name = ""
    
    if request.method == "POST":
        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                error_message, predicted_job, extracted_skills, user_name = process_resume(file)
                if not error_message:
                    missing_skills = compare_skills(predicted_job, extracted_skills, user_name)
                    try:
                        conn_resume = get_db_connection("resume_db")
                        cursor_resume = conn_resume.cursor()
                        skills_str = ", ".join(extracted_skills)
                        cursor_resume.execute("INSERT INTO resumes (name, skills) VALUES (%s, %s)", (user_name, skills_str))
                        conn_resume.commit()
                        cursor_resume.close()
                        conn_resume.close()
                    except Exception as db_error:
                        error_message = f"[ERROR] Database error: {db_error}"
    
    return render_template("index.html", 
                           predicted_job=predicted_job or "", 
                           error_message=error_message or "", 
                           extracted_skills=extracted_skills,
                           missing_skills=missing_skills,
                           recommended_skills=missing_skills)

if __name__ == "__main__":
    app.run(debug=True)
