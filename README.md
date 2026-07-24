# 🎓 Graduate Admission Score Prediction System

## 📖 Overview
A Graduate Admission Score Prediction System built with Streamlit and Artificial Neural Networks (ANN). The system predicts admission chances based on academic profile factors like GRE score, TOEFL score, CGPA, research experience, and more.


---
## ✨ Features
- 🎓 Score Prediction - Predicts admission chance score (0-100%) using ANN
- 📊 Profile Analysis - Analyzes student profile with detailed insights
- 💡 Improvement Suggestions - Personalized recommendations to improve admission chances
- 📈 Interactive Visualization - Visual comparison with average scores
- 🎯 Risk Assessment - Quick risk level indicator
- 📋 Detailed Reporting - Comprehensive results with confidence metrics


---
## 🛠️ Technologies Used
- Python 3.10+ - Core programming language
- TensorFlow/Keras - Deep learning framework
- Streamlit - Web application framework
- Scikit-learn - Data preprocessing and scaling
- Pandas/NumPy - Data manipulation
- Matplotlib/Seaborn - Data visualization


---
## 📁 Project Structure

```
graduate-admission-prediction-system/
├── app.py                    # Main Streamlit application (UI)
├── predict.py                # Prediction functions
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── .gitignore               # Git ignore file
├── data/                     # Dataset (gitignored)
│   └── Admission_Predict_Ver1.1.csv
├── models/                   # Trained model files (gitignored)
│   ├── admission_model.h5
│   ├── scaler.pkl
│   └── feature_names.pkl
├── saved_visualizations/     # Generated plots (gitignored)
│   ├── eda_admission.png
│   ├── admission_training_history.png
│   └── predictions_vs_actual.png
└── README.md                 # Project documentation
```

---
## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/anjalitarkar101/graduate-admission-score-prediction-system.git
cd graduate-admission-prediction
```

### Step 2: Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```
This will:
- Create required directories (data/, models/, saved_visualizations/)
- Install all dependencies

### Step 3: Download Dataset
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/mohansacharya/graduate-admissions
- **Name:** Graduate Admission 2
- **File:** Admission_Predict_Ver1.1.csv
- **Size:** ~30 KB
- **Columns:** 9

After downloading, place the file in the `data/` folder:
```
data/
└── Admission_Predict_Ver1.1.csv
```

### Step 4: Train the Model
```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the ANN model
- Save the model and artifacts to models/ folder

### Step 5: Run the Application
```bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501
```

---
## 📊 How It Works
1. Data Preprocessing
- Loads admission dataset
- Handles missing values
- Scales features using MinMaxScaler
- Splits data into train/test sets

2. Model Architecture (ANN)

```
Input Layer (7 features)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Dense Layer (16 neurons, ReLU)
    ↓
Output Layer (1 neuron, Linear)
    ↓
Prediction (Admission Score)
```

3. Prediction Pipeline
- User enters academic profile
- Input is preprocessed and scaled
- Model predicts admission score
- Results displayed with analysis


---
## 🔧 Dependencies

Create a `requirements.txt` file with the following content:

```txt
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
streamlit==1.50.0
```

Install using:

```bash
pip install -r requirements.txt
```


---
## 📝 Usage Guide
1. Enter Your Profile - Fill in academic details (GRE, TOEFL, CGPA, etc.)
2. Click Predict - Press the "Predict Admission Score" button
3. View Results - See your admission chance score with interpretation
4. Get Suggestions - Review personalized improvement suggestions
5. Compare - See how you compare to the average applicant


---
## 📊 Dataset Information

| Feature | Description |
|:--------|:------------|
| **GRE Score** | Graduate Record Examination score (260-340) |
| **TOEFL Score** | Test of English as Foreign Language score (0-120) |
| **University Rating** | University rating (1-5) |
| **SOP** | Statement of Purpose strength (1-5) |
| **LOR** | Letter of Recommendation strength (1-5) |
| **CGPA** | Cumulative Grade Point Average (0-10) |
| **Research** | Research experience (0 or 1) |
| **Chance of Admit** | Target variable (0-1) |


---
## 📄 License
This project is licensed under the MIT License.

© 2026 Anjali Tarkar. All rights reserved.


---
## 👩‍💻 Author
**Anjali Tarkar**
- GitHub: https://github.com/anjalitarkar101
- Email: anjalitarkar101@gmail.com


---
## ⭐ Show Your Support
If you find this project useful, please give it a star on GitHub!


----
## 🙏 Acknowledgments
- Mohan S Acharya- For the Graduate Admission 2  Dataset 
- TensorFlow/Keras - For the deep learning framework
- Streamlit - For the awesome web framework


