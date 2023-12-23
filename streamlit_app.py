import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

data = pd.read_csv('medical_data.csv')

data = data.dropna()

# Combine Symptoms, Causes, and Gender columns to create input features
data['Combined'] = data['Symptoms'] + ' ' + data['Causes'] + ' ' + data['Gender']


# Text vectorization using CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit_transform(data['Combined'])

# Load the model
disease_classifier = pickle.load(open('disease_classifier.pkl', 'rb'))
medicine_classifier = pickle.load(open('medicine_classifier.pkl', 'rb'))


# Streamlit app
def main():
    st.title('MediPredict: AI-Driven Symptom Diagnosis')
    st.write('Enter your symptoms below:')

    # Get user input
    user_symptoms = st.text_input('Symptoms')
    user_symptoms = user_symptoms.split(',')
    user_symptoms = ' '.join(user_symptoms)
    symptoms_input = vectorizer.transform([user_symptoms])

    if st.button('Predict'):
        # Predict the disease and medicine based on the symptoms
        predicted_disease = disease_classifier.predict(symptoms_input)
        predicted_medicine = medicine_classifier.predict(symptoms_input)

        # Output the results
        st.write(f'Predicted disease: {predicted_disease}')
        st.write(f'Predicted medicine: {predicted_medicine}')
if __name__ == '__main__':
    main()
