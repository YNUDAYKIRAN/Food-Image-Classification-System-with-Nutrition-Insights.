# Food-Image-Classification-System-with-Nutrition-Insights.


🚀 Overview

This project is an end-to-end deep learning-based computer vision system that classifies food images and provides real-time nutrition insights.
Unlike basic classifiers, this system goes beyond prediction by linking results with practical health-related data, making it useful for real-world applications.

# 🎯 Objectives

Classify food images into multiple categories
Provide real-time predictions with confidence scores
Integrate nutrition data for each predicted food item
Build a deployable full-stack AI application

# 🧠 Models Used

Custom Convolutional Neural Network (CNN),
VGG16 (Transfer Learning),
ResNet (Deep Residual Network).

# ⚙️ Tech Stack

Backend: Flask
Frontend: HTML, CSS, JavaScript
Database/Cache: Redis
Libraries: TensorFlow / Keras, NumPy, OpenCV, Matplotlib

# 🔄 Workflow

1. Image Input (User Upload)
2. Image Preprocessing & Augmentation
3. Model Prediction
4. Confidence Score Generation
5. Nutrition Data Retrieval (via Redis)
6. Display Results on Web Interface


# 📊 Features

1. Multi-class classification (34 food categories)
2. Real-time predictions
3. Confidence score display
4. Nutrition insights integration
5. Interactive web interface
6. Model comparison (CNN vs VGG16 vs ResNet)


# 📈 Model Evaluation
1. Accuracy
2. Confusion Matrix
3. Classification Report (Precision, Recall, F1-score)


# 🖥️ Demo

👉 Live / Project Link


# 📂 Project Structure

├── static/              # CSS, JS, images

├── templates/           # HTML files

├── models/              # Saved trained models

├── app.py               # Flask application

├── utils.py             # Helper functions

├── requirements.txt     # Dependencies

└── README.md


# ⚡ Installation & Setup

1️. Clone the Repository
git clone https://github.com/your-username/food-image-classification.git
cd food-image-classification

2️. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

3️. Install Dependencies
pip install -r requirements.txt

4️. Run Redis Server
Make sure Redis is installed and running:
redis-server

5️. Run the Application
python app.py

Open in browser:
http://127.0.0.1:5000/

# 🧪 Example Output

Predicted Class: Pizza
Confidence: 92.4%
Calories: 285 kcal per slice
Protein / Carbs / Fats breakdown


# 🔥 Key Learnings
1. Deep understanding of CNN architectures
2. Hands-on experience with transfer learning
3. Model evaluation beyond accuracy
4. Integrating AI models into full-stack applications
5. Using Redis for performance optimization



# 📬 Contact

If you found this useful or want to collaborate:

📧 Your Email : udaykiranyerranaga@gmail.com

🔗 LinkedIn Profile : https://www.linkedin.com/in/mr-uday-kiran-42803a2b3/
