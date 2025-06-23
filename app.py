import os
from flask import Flask, render_template, request, jsonify
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from pdfminer.high_level import extract_text

app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text_from_pdf(file_path):
    return extract_text(file_path)

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def calculate_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files or 'jobDescription' not in request.form:
        return jsonify({'error': 'Missing resume or job description'}), 400

    resume_file = request.files['resume']
    job_description = request.form['jobDescription']

    if resume_file.filename == '':
        return jsonify({'error': 'No resume selected'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
    resume_file.save(file_path)

    try:
        if file_path.endswith('.docx'):
            resume_text = extract_text_from_docx(file_path)
        elif file_path.endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_path)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        processed_resume = preprocess_text(resume_text)
        processed_job = preprocess_text(job_description)
        
        similarity_score = calculate_similarity(processed_resume, processed_job)
        
        # Extract key skills from job description
        job_doc = nlp(job_description)
        key_skills = [ent.text for ent in job_doc.ents if ent.label_ in ['SKILL', 'ORG', 'PRODUCT']]
        
        # Find matching skills in resume
        resume_doc = nlp(resume_text)
        resume_skills = [ent.text for ent in resume_doc.ents if ent.label_ in ['SKILL', 'ORG', 'PRODUCT']]
        
        matching_skills = list(set(key_skills) & set(resume_skills))

        os.remove(file_path)  # Clean up uploaded file

        return jsonify({
            'score': similarity_score,
            'matching_skills': matching_skills,
            'missing_skills': list(set(key_skills) - set(resume_skills))
        })

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)