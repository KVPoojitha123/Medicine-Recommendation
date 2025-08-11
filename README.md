import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

try:
    training_df = pd.read_csv('/content/Training.csv') 
    medications_df = pd.read_csv('/content/medications.csv') 
    description_df = pd.read_csv('/content/description.csv') 
    diets_df = pd.read_csv('/content/diets.csv')  
    symptoms_df = pd.read_csv('/content/symtoms_df.csv')  

    X = training_df.drop(columns=['prognosis']) 
    y = training_df['prognosis']  

    X = X.iloc[:, :15]  

    le = LabelEncoder()
    Y = le.fit_transform(y)  
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)  # Increased test size

    np.random.seed(42)
    X_train += np.random.normal(0, 0.05, X_train.shape)  

    model = LogisticRegression(max_iter=200, C=0.3)  
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    disease_to_medications = {row['Disease']: row['Medication'] for _, row in medications_df.iterrows()}
    disease_to_description = {row['Disease']: row['Description'] for _, row in description_df.iterrows()}
    disease_to_diet = {row['Disease']: row['Diet'] for _, row in diets_df.iterrows()}

    symptoms_dict = {symptom: idx for idx, symptom in enumerate(X.columns)}

    def predict_disease(patient_symptoms):
        """Predict disease based on provided symptoms."""
        input_vector = np.zeros(len(symptoms_dict))

        # Convert symptoms into feature vector
        for symptom in patient_symptoms:
            symptom = symptom.lower().strip()
            if symptom in symptoms_dict:
                input_vector[symptoms_dict[symptom]] = 1
            else:
                print(f"Symptom '{symptom}' not recognized!")

        input_vector = input_vector.reshape(1, -1)

        # Predict the disease using the trained model
        prediction_index = model.predict(input_vector)[0]
        predicted_disease = le.inverse_transform([prediction_index])[0]  # Decode disease

        return predicted_disease

    def recommend_medicine(predicted_disease):
        """Recommend medication based on the predicted disease."""
        medications = disease_to_medications.get(predicted_disease, "No medication found")
        return medications

    def get_disease_info(predicted_disease):
        """Retrieve disease description and diet based on the predicted disease."""
        description = disease_to_description.get(predicted_disease, "No description available")
        diet = disease_to_diet.get(predicted_disease, "No diet recommendations")
        return description, diet

    # Prompt user for symptoms
    symptoms_input = input("Enter your symptoms separated by commas (e.g., itching, skin_rash): ")
    patient_symptoms = [symptom.strip() for symptom in symptoms_input.split(',')]

    # Predict the disease based on the symptoms
    predicted_disease = predict_disease(patient_symptoms)
    print(f"\nPredicted Disease: {predicted_disease}")

    # Recommend medication based on the predicted disease
    recommended_meds = recommend_medicine(predicted_disease)
    print(f"Recommended Medications: {recommended_meds}")

    # Get disease description and diet
    disease_description, disease_diet = get_disease_info(predicted_disease)
    print(f"Disease Description: {disease_description}")
    print(f"Diet Recommendations: {disease_diet}")

except Exception as e:
    print(f"An error occurred: {e}")
