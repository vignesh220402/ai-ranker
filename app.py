import os
import re
from flask import Flask, render_template, request, jsonify
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from pdfminer.high_level import extract_text
import openai
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')
# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# Common skills dictionary with categories
SKILL_CATEGORIES = {
    'programming_languages': [
        'python', 'java', 'javascript',"c++","c"
    ],
    'web_technologies': [
        "html","css","javascript"
    ],
    'databases': [
        'sql','mysql', 'postgresql', 'mongodb','sqlite','mariadb','firebase'
    ],
    'cloud_platforms': [
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab',
        'github actions', 'circleci', 'ansible', 'puppet', 'chef'
    ],
    'machine_learning': [
        'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'opencv', 'pandas', 'numpy',
        'matplotlib', 'seaborn', 'nltk', 'spacy', 'bert', 'transformers'
    ],
    'soft_skills': [
        "web development","front-end","backend","flutter","react native"
    ],
    'courses_certifications': [
        "embedded system","iot","data science","microprocessor","controlsystem","consumer electronics"
    ]
}


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text
def extract_text_from_pdf(file_path):
    return extract_text(file_path)
def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()
def analyze_with_gpt(text, context='resume'):
    try:
        prompt = f"""Analyze this {context} and extract the following information in JSON format:
        1. All technical skills and tools mentioned
        2. Years of experience for each skill (if mentioned)
        3. Key achievements and responsibilities
        4. Education and certifications
        5. Soft skills demonstrated
        Text to analyze: {text}
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are an expert resume analyzer. Extract and categorize information accurately."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.3
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"GPT Analysis Error: {str(e)}")
        return None
def extract_skills(text):
    text = text.lower()
    skills = set()
    # Use GPT to extract skills
    gpt_analysis = analyze_with_gpt(text)
    if gpt_analysis:
        try:
            import json
            analysis_data = json.loads(gpt_analysis)
            if 'technical_skills' in analysis_data:
                skills.update(analysis_data['technical_skills'])
            if 'soft_skills' in analysis_data:
                skills.update(analysis_data['soft_skills'])
            if 'relevant_courses_and_certifications' in analysis_data:
                skills.update(analysis_data['relevant_courses_and_certifications'])
        except json.JSONDecodeError:
            pass
    # Extract skills from predefined categories
    for category, category_skills in SKILL_CATEGORIES.items():
        for skill in category_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                skills.add(skill)
    # Extract experience levels and years
    experience_patterns = [
        r'\b(\d+)\+?\s*(?:year|yr)s?\s+(?:of\s+)?experience\s+(?:in|with)?\s+([^.,:;]+)',
        r'(?:experienced?|skilled|proficient)\s+in\s+([^.,:;]+)',
        r'\b([^.,:;]+)\s+(?:developer|engineer|specialist|expert)\b'
    ]
    for pattern in experience_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if len(match.groups()) > 1:
                years, skill = match.groups()
                skills.add(f"{skill.strip()} ({years}+ years)")
            else:
                skill = match.group(1).strip()
                if len(skill) > 2:
                    skills.add(skill)
    # Extract technical terms using spaCy
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT']:
            skills.add(ent.text.lower())
    return list(skills)

def calculate_similarity(resume_text, job_description):
    # Clean and preprocess texts
    resume_text_clean = clean_text(resume_text)
    job_description_clean = clean_text(job_description)
    # Get GPT analysis for both resume and job description
    resume_analysis = analyze_with_gpt(resume_text, 'resume')
    job_analysis = analyze_with_gpt(job_description, 'job description')
    
    # Calculate TF-IDF similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text_clean, job_description_clean])
    content_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    # Extract skills
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description))
    
    # Calculate skill matches with categories
    matching_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills
    
    # Calculate category-based skill coverage
    category_weights = {
        'programming_languages': 0.25,
        'web_technologies': 0.2,
        'databases': 0.15,
        'cloud_platforms': 0.15,
        'machine_learning': 0.15,
        'soft_skills': 0.1,
        'courses_certifications': 0.1
    }
    
    category_scores = []
    for category, weight in category_weights.items():
        category_skills = set(SKILL_CATEGORIES[category])
        required_skills = category_skills.intersection(job_skills)
        if required_skills:
            matched_skills = category_skills.intersection(matching_skills)
            category_score = len(matched_skills) / len(required_skills) * weight
            category_scores.append(category_score)
    
    # Calculate semantic similarity using GPT analysis
    semantic_score = 0
    if resume_analysis and job_analysis:
        try:
            resume_data = json.loads(resume_analysis)
            job_data = json.loads(job_analysis)
            
            # Compare education requirements
            if 'education' in resume_data and 'education' in job_data:
                education_match = any(edu.lower() in str(resume_data['education']).lower() 
                                    for edu in job_data['education'])
                semantic_score += 0.1 if education_match else 0
            
            # Compare achievements and responsibilities
            if 'achievements' in resume_data and 'responsibilities' in job_data:
                resp_matches = sum(1 for resp in job_data['responsibilities'] 
                                if any(achievement.lower() in resp.lower() 
                                    for achievement in resume_data['achievements']))
                semantic_score += min(0.2, resp_matches * 0.05)
        except json.JSONDecodeError:
            pass
    
    # Calculate weighted skill score
    skill_score = sum(category_scores) if category_scores else 0
    
    # Calculate final weighted score
    content_weight = 0.2
    skill_weight = 0.5
    semantic_weight = 0.3
    final_score = (content_similarity * content_weight + 
                  skill_score * skill_weight + 
                  semantic_score * semantic_weight) * 100
    
    return round(final_score, 2), list(matching_skills), list(missing_skills)

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

        score, matching_skills, missing_skills = calculate_similarity(resume_text, job_description)

        os.remove(file_path)  # Clean up uploaded file

        return jsonify({
            'score': score,
            'matching_skills': matching_skills,
            'missing_skills': missing_skills
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)