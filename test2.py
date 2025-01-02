from pdfminer.high_level import extract_text
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Load the CSV file with CV data
resume_file_path = "/Users/diasalehs/Desktop/Resume.csv"
data = pd.read_csv(resume_file_path)

# Select the relevant columns
X = data['Resume_str']  # The text content of the resumes
y = data['Category']  # The corresponding categories (labels)

# Check the distribution of categories
print(y.value_counts())

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the resume text to create feature vectors
X_vectors = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=1))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Function to predict the category of a job description
def predict_job_category(job_description, model, vectorizer):
    # Transform the job description using the vectorizer
    job_vector = vectorizer.transform([job_description])

    # Predict the category using the trained model
    predicted_category = model.predict(job_vector)
    return predicted_category[0], job_vector

# Function to match CVs to a job description and return the top N matches
def match_cvs_to_job(job_description, data, model, vectorizer, top_n=3):
    # Predict the job category
    job_category, job_vector = predict_job_category(job_description, model, vectorizer)

    # Filter CVs that match the predicted job category
    matched_cvs = data[data['Category'] == job_category]

    # Transform matched CVs to vectors
    matched_cv_vectors = vectorizer.transform(matched_cvs['Resume_str'])

    # Calculate cosine similarities between the job description and matched CVs
    similarities = cosine_similarity(job_vector, matched_cv_vectors).flatten()

    # Get indices of the top N matching CVs
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Return the top N matching CVs along with their similarity scores
    top_cvs = matched_cvs.iloc[top_indices]
    top_scores = similarities[top_indices]

    return top_cvs, top_scores

# Example job description (as text)
job_description = """
Graphic designers use their creativity, technical skills, and design expertise to craft compelling visual concepts that resonate with a target audience. They specialize in creating visual designs for marketing materials, social media, websites, and product packaging. Collaborating with team members, including copywriters and marketing teams, graphic designers ensure that all materials align with brand guidelines and effectively communicate the intended message.

This role requires proficiency in design software such as Photoshop, InDesign, and Adobe Creative Suite, along with a strong understanding of design principles, typography, and color theory. With excellent communication skills and adaptability, graphic designers play a vital role in meeting deadlines and exceeding project goals.


"""

# Find matching CVs for the job description
matched_cvs, scores = match_cvs_to_job(job_description, data, model, vectorizer)

# Display the matched CVs
print(f'Matched CVs for job category "{predict_job_category(job_description, model, vectorizer)[0]}":')
for index, row in matched_cvs.iterrows():
    print(f'ID: {row["ID"]}, Category: {row["Category"]}, Similarity Score: {scores[np.where(matched_cvs.index == index)[0][0]]:.4f}')
