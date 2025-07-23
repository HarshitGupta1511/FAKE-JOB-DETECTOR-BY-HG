import gradio as gr
import pickle
import os
from fpdf import FPDF
import tempfile

# Load models and vectorizer
model_names = ["randomforest", "naivebayes", "svm", "decisiontree"]
models = {}
for name in model_names:
    with open(f"model_files/{name}.pkl", "rb") as f:
        models[name] = pickle.load(f)

with open("model_files/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model_files/accuracies.pkl", "rb") as f:
    accuracies = pickle.load(f)

def preprocess_input(title, location, description, benefits):
    # Clean up each input to remove leading/trailing spaces
    title = title.strip().lower()
    location = location.strip().lower()
    description = description.strip().lower()
    benefits = benefits.strip().lower()

    return f"{title} {location} {description} {benefits}"

def clean_for_pdf(text):
    # Remove problematic unicode characters
    return (
        text.replace("–", "-")
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("•", "-")
            .replace("🔴", "")
            .replace("🟢", "")
            .replace("📝", "")
            .replace("📍", "")
            .replace("📄", "")
            .replace("🎁", "")
            .encode("latin-1", "replace")
            .decode("latin-1")
    )

def predict_and_generate_pdf(title, location, description, benefits):
    import numpy as np
    from fpdf import FPDF
    import tempfile

    # Validate input
    if not all(field.strip() for field in [title, location, description, benefits]):
        return "⚠️ Please fill all fields properly before prediction.", None

    input_text = preprocess_input(title, location, description, benefits)
    vect = vectorizer.transform([input_text])

    results = {}
    results_no_emoji = {}
    for name, model in models.items():
        pred = int(model.predict(vect)[0])
        label = "🟢 Real" if pred == 0 else "🔴 Fake"
        plain_label = "Real" if pred == 0 else "Fake"
        acc = round(float(accuracies[name]) * 100, 2)
        results[name] = f"{label} ({acc}%)"
        results_no_emoji[name] = f"{plain_label} ({acc}%)"

    # PDF Generation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, clean_for_pdf("Fake Job Detection Report"), ln=True, align="C")
    pdf.ln(10)

    # Add job details
    pdf.multi_cell(0, 10, clean_for_pdf(
        f"Job Title: {title}\nLocation: {location}\n\nDescription:\n{description}\n\nBenefits:\n{benefits}\n"
    ))

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, clean_for_pdf("Model Predictions:"), ln=True)
    pdf.set_font("Arial", size=12)

    for model_name, result in results_no_emoji.items():
        pdf.cell(200, 10, clean_for_pdf(f"{model_name.upper()}: {result}"), ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)

    output_str = "\n".join([f"{k.upper()}: {v}" for k, v in results.items()])
    return output_str, tmp.name

# Gradio UI
with gr.Blocks(title="Fake Job Detector") as demo:
    gr.Markdown("## 🕵️‍♂️ Fake Job Detection App by Harshit")
    gr.Markdown("Enter job details to see predictions from ML models.")

    with gr.Row():
        title = gr.Textbox(label="📝 Job Title")
        location = gr.Textbox(label="📍 Location")

    description = gr.Textbox(lines=5, label="📄 Job Description")
    benefits = gr.Textbox(lines=3, label="🎁 Benefits")

    predict_btn = gr.Button("🚀 Predict")
    output = gr.Textbox(label="🔍 Predictions from ML Models", lines=8)
    download = gr.File(label="📥 Download Prediction Report (PDF)", interactive=False)

    predict_btn.click(fn=predict_and_generate_pdf,
                      inputs=[title, location, description, benefits],
                      outputs=[output, download])

    # 🔁 FULL list of examples (restored)
    gr.Examples(
        examples=[
            ["Software Engineer", "San Francisco, CA",
             "We are looking for a skilled engineer to work on backend systems using Python and cloud technologies.",
             "Health insurance, Paid time off, Remote work option"],

            ["Data Entry Clerk", "Remote",
             "Urgent opening! Earn $500/day by filling out forms. No experience needed. Click links to start immediately.",
             "Huge daily payout, Free laptop, Quick joining bonus"],

            ["Marketing Manager", "London, UK",
             "Lead digital marketing campaigns across SEO, SEM, and social platforms. Collaborate with global teams.",
             "Flexible hours, Work from home, Career development"],

            ["Administrative Assistant", "Toronto, ON",
             "Hiring now! Simple tasks, no skills required. Send your personal info and we’ll contact you ASAP.",
             "Amazing rewards, Free phone, Instant hiring"],

            ["Senior Software Engineer", "San Francisco, CA",
             "Develop and maintain scalable applications.",
             "401k, health insurance"],

            ["HR Assistant", "Remote",
             "Assist with hiring and onboarding.",
             "Flexible hours, remote work"],

            ["Data Analyst", "New York, NY",
             "Analyze data trends and prepare reports.",
             "Free lunch, gym membership"],

            ["Work from Home", "Unknown",
             "Earn money by completing simple tasks online. No experience needed.",
             "Daily payouts, quick signup"]
        ],
        inputs=[title, location, description, benefits],
        label="🧪 Try These Examples"
    )

demo.launch()
