# 🕵️‍♂️ Fake Job Detector by HG


A Machine Learning-based web application that detects whether a job posting is **real** or **fake** using NLP and classification models. It is designed to help users avoid fraudulent job listings by analyzing various textual and categorical inputs.

---

## 🚀 Live Demo

🔗 **Try it here**: [Fake Job Detector on Hugging Face](https://huggingface.co/spaces/harshitg15/real-and-fake-job-detector)

---


## 🔍 How It Works

1. **User inputs** job details such as title, description, location, requirements, etc.
2. Text data is **preprocessed** and **vectorized**.
3. Multiple **ML models** (Random Forest, XGBoost, etc.) analyze the input.
4. The app displays the **predicted result** — *Real* or *Fake* — along with model accuracies.

---

## 📦 Tech Stack

| Layer          | Technology                             |
|----------------|------------------------------------------|
| Interface      | Gradio                                  |
| ML Models      | Scikit-learn, XGBoost, joblib            |
| Language       | Python                                  |
| Deployment     | Hugging Face Spaces                      |
| Dataset        | [Fake Job Postings (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) |

---

## 🧠 Models Used

- ✅ Random Forest Classifier
- ✅ SVM Classifier
- ✅ Naive Bayes Classifier
- ✅ Decision Tree Classifier 


Each model is trained on a cleaned version of the dataset and serialized using `joblib` for fast predictions.

---

## 📁 Project Structure

```bash
FAKE-JOB-DETECTOR-BY-HG/
│
├── app.py                 # Gradio app entry point
├── utils.py               # (Optional) Preprocessing or utility functions
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── dataset/               # Dataset used (optional inclusion)
