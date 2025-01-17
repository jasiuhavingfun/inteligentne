import os
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Configure the static folder for uploads
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
model = load_model("plant_disease_model.h5")

# Class labels (based on your dataset structure)
class_labels = ["Healthy", "Powdery", "Rust"]

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if the file is uploaded
        if "file" not in request.files:
            return "No file uploaded!", 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return "No file selected!", 400
        
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            
            # Process the image
            img = load_img(filepath, target_size=(150, 150))  # Resize to model input size
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Predict the class
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]
            
            # Relative path for the template to access
            relative_filepath = os.path.join("static", "uploads", file.filename)
            
            return render_template("result.html", 
                                   image_path=relative_filepath, 
                                   prediction=predicted_class)
    
    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create upload directory if not exists
    app.run(debug=True)
