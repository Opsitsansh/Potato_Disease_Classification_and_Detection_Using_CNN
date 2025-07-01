import os
import tensorflow as tf
from PIL import Image
import numpy as np
from google.cloud import storage
from flask import jsonify

# Constants
BUCKET_NAME = "ansh-tf-models"
MODEL_PATH_GCS = "models/potatoes_saved_model/"
MODEL_PATH_LOCAL = "/tmp/potatoes_saved_model"
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Global model instance
model = None

# Function to download model folder from GCS
def download_blob_folder(bucket_name, prefix, destination_dir):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        rel_path = blob.name[len(prefix):]
        if rel_path:
            local_path = os.path.join(destination_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")

# Main Cloud Function handler
def predict(request):
    global model

    # Load model from GCS if not already loaded
    if model is None:
        try:
            print("Downloading model from GCS...")
            download_blob_folder(BUCKET_NAME, MODEL_PATH_GCS, MODEL_PATH_LOCAL)
            model = tf.saved_model.load(MODEL_PATH_LOCAL)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return jsonify({"error": "Model load failed", "details": str(e)}), 500

    # Check for uploaded file
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    image_file = request.files["file"]

    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process the image
        img = Image.open(image_file).convert("RGB").resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0  # ✅ Convert to float32
        img_array = tf.expand_dims(img, 0)

        # Perform inference using SavedModel signature
        infer = model.signatures["serve"]
        output = infer(tf.constant(img_array))
        predictions = list(output.values())[0].numpy()

        # Format result
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)

        return jsonify({
            "class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


