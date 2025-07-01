=> Description
• This project leverages Convolutional Neural Networks (CNN) to classify potato leaf diseases into categories such as Early Blight, Late Blight, and Healthy. It aims to help farmers and researchers detect diseases early using image-based analysis, enhancing agricultural productivity and reducing crop loss.

=> Problem Statement
• Potato crops are vulnerable to various diseases that significantly reduce yield. Manual detection is time-consuming and error-prone. This project automates the detection process using deep learning techniques for faster and more accurate diagnosis.

=> Features
• Image classification using CNN
• Trained on the PlantVillage dataset
• Achieves high accuracy in disease detection
• Frontend in React, backend in FastAPI
• TensorFlow Serving is used for model deployment

=> Tech Stack
• Python, TensorFlow, Keras
• FastAPI (Backend)
• React.js (Frontend)
• Google Cloud Storage & TensorFlow Serving
• Jupyter Notebook (Model Training)

=> Dataset
• Source: PlantVillage Dataset (Potato Leaves)
  • Classes:
   • Early Blight
   • Late Blight
   • Healthy

==>> HOW TO RUN

=> Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

=> Frontend(React)
cd frontend
npm install
npm start

=>Model Inference
•  Upload an image via the UI or test using Postman
•  Model returns predicted class: Early Blight, Late Blight, or Healthy

=> Model Accuracy
• Training Accuracy: 98.5%
• Validation Accuracy: 96.3%
• Confusion Matrix & loss/accuracy graphs available in notebooks/
 
