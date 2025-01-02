import numpy as np
from pdfminer.high_level import extract_text
from .models import Application

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def match_cvs_to_job(job_description, job_position, model, vectorizer, top_n=5):
    # Get all applications for the job position
    applications = Application.objects.filter(job_position=job_position)

    if not applications.exists():
        return "No applicants have applied for this job position.", []

    # Extract text from each CV
    cv_texts = []
    for application in applications:
        cv_path = application.cv.path
        cv_text = extract_text_from_pdf(cv_path)
        cv_texts.append(cv_text)

    # Vectorize the job description and CV texts

    job_category = predict_job_category(job_description, model, vectorizer)[0]
    print(job_category)
    job_vector = vectorizer.transform([job_category])
    cv_vectors = vectorizer.transform(cv_texts)

    # Calculate similarities between the job description and CVs
    similarities = np.dot(cv_vectors, job_vector.T).toarray().flatten()

    # Normalize similarities to a percentage
    similarities = similarities * 100

    # Get indices of the top N matching CVs
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Convert indices to a list of primary keys
    application_ids = list(applications.values_list('id', flat=True))
    top_application_ids = [application_ids[i] for i in top_indices]

    # Get the top N matching applications
    top_applications = [applications.get(id=app_id) for app_id in top_application_ids]
    top_scores = similarities[top_indices]

    return top_applications, top_scores



def predict_job_category(job_description, model, vectorizer):
    job_vector = vectorizer.transform([job_description])
    return model.predict(job_vector)