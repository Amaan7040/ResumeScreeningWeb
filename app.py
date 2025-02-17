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

# Define a set of common skills (in lowercase)
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

# ---------------- Database Connection ----------------


def get_db_connection():
    """
    Establishes and returns a connection to the MySQL database.
    Replace the host, user, password, and database with your own settings.
    """
    connection = mysql.connector.connect(
        host="localhost",  # Your MySQL host
        user="root",  # Your MySQL username
        password="amaan@khan704093",  # Your MySQL password
        database="resume_db", # Your MySQL database name
        auth_plugin="mysql_native_password"
    )
    return connection


# ---------------- Resume Processing Functions ----------------


def extract_text_from_file(file):
    text = ""
    if file.filename.endswith(".pdf"):
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
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
    else:
        print("[ERROR] Unsupported file format!")
        return None

    return text.strip() if text.strip() else None


def extract_sections(text):
    sections = {
        "summary": None,
        "education": None,
        "work_experience": None,
        "projects": None,
        "skills": None,
        "certifications": None,
        "publications": None,
        "competencies": None,
    }

    section_patterns = {
        "summary": r"(summary|profile|about me)[:\n]",
        "education": r"(education|academic background)[:\n]",
        "work_experience":
        r"(work experience|employment history|professional experience)[:\n]",
        "projects": r"(projects|personal projects|academic projects)[:\n]",
        "skills": r"(skills|technical skills|programming languages)[:\n]",
        "certifications": r"(certifications|courses|training)[:\n]",
        "publications": r"(publications|research papers)[:\n]",
        "competencies": r"(competencies|key competencies|expertise)[:\n]",
    }

    for section, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_idx = match.end()
            next_match = min([
                m.start() for m in [
                    re.search(p, text[start_idx:], re.IGNORECASE)
                    for p in section_patterns.values()
                ] if m
            ],
                             default=len(text))
            sections[section] = text[start_idx:start_idx + next_match]

    return sections


def extract_skills(text):
    extracted_skills = set()
    doc = nlp(text)

    for token in doc:
        word = token.text.lower()  # Convert token to lowercase
        if word in common_skills:
            extracted_skills.add(word)

    return list(extracted_skills)


def extract_name(text):
    name = ""
    lines = text.split('\n')
    if lines:
        name = lines[0].strip()  # Assume the first line contains the name
        return name
    else:
        return None


def load_model_and_vectorizer():
    model_path = "model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, "rb") as model_file:
            rf = pickle.load(model_file)
        with open(vectorizer_path, "rb") as vectorizer_file:
            tfidf = pickle.load(vectorizer_file)
        print("[INFO] Model and Vectorizer loaded successfully.")
        return rf, tfidf
    else:
        print("[ERROR] Model or Vectorizer missing!")
        return None, None


def process_resume(file):
    rf, tfidf = load_model_and_vectorizer()
    if rf is None or tfidf is None:
        return "[ERROR] ML model is missing!", None, None, None

    text = extract_text_from_file(file)
    if not text:
        return "[ERROR] Invalid or unsupported file format!", None, None, None

    user_name = extract_name(text)
    extracted_skills = extract_skills(text)
    extract_section = extract_sections(text)

    try:
        text_vectorized = tfidf.transform([text])
        print("Vectorized Input Shape:", text_vectorized.shape)
        predicted_job = rf.predict(text_vectorized)[0]
        print("Predicted Job:", predicted_job)

        return None, predicted_job, extracted_skills, extract_section, user_name
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return f"[ERROR] Prediction failed: {e}", None, extracted_skills, extract_section, user_name


# ---------------- Flask Application ----------------

app = Flask(__name__, template_folder="templates")


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_job = None
    error_message = None
    extracted_skills = []
    extract_section = {}
    user_name = ""

    if request.method == "POST":
        # Get user's name from the form (make sure your index.html has a 'name' input)

        if "resume" not in request.files:
            error_message = "No file uploaded!"
        else:
            file = request.files["resume"]
            if file.filename == "":
                error_message = "No selected file!"
            else:
                error_message, predicted_job, extracted_skills, extract_section, user_name = process_resume(
                    file)

                # If there was no error in processing, save name and skills to the database.
                if not error_message:
                    try:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        # Convert the list of skills to a comma-separated string.
                        skills_str = ", ".join(extracted_skills)
                        insert_query = "INSERT INTO resumes (name, skills) VALUES (%s, %s)"
                        cursor.execute(insert_query, (user_name, skills_str))
                        conn.commit()
                        cursor.close()
                        conn.close()
                        print("[INFO] User data saved to database.")
                    except Exception as db_error:
                        error_message = f"[ERROR] Database error: {db_error}"
                        print(error_message)

    return render_template("index.html",
                           predicted_job=predicted_job or "",
                           error_message=error_message or "",
                           extracted_skills=extracted_skills,
                           extract_section=extract_section)


if __name__ == "__main__":
    app.run(debug=True)