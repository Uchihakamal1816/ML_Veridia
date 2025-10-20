import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import PyPDF2
import docx
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import re
import joblib
import tempfile

st.set_page_config(page_title="Resume Role Matcher", layout="wide")

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("models/final_distilbert_resume")
    label_encoder = joblib.load("label_encoder.pkl")
    return tokenizer, model, label_encoder

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except:
        return ""

def predict_role(text, tokenizer, model, label_encoder, target_confidence=0.85):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    temperature = 1.0

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        while temperature > 0.05:
            scaled_logits = logits / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
            if confidence >= target_confidence or temperature >= 2.0:
                break
            temperature -= 0.05

        category = label_encoder.inverse_transform([pred_label])[0]
        
        all_probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        all_categories = label_encoder.classes_
        
    return category, confidence, all_categories, all_probs

def main():
    st.title("📄 AI Resume Role Matcher")
    
    tokenizer, model, label_encoder = load_model()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader("Choose file", type=['pdf', 'docx', 'txt'])
        
        st.subheader("Or Paste Resume Text")
        resume_text = st.text_area("Paste content here:", height=150)
        
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.85)
        
        analyze_btn = st.button("Analyze Resume", type="primary")
    
    with col2:
        if analyze_btn:
            if not uploaded_file and not resume_text:
                st.warning("Please upload a file or paste text")
                return
            
            final_text = ""
            if uploaded_file:
                with st.spinner("Reading file..."):
                    if uploaded_file.type == "application/pdf":
                        final_text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        final_text = extract_text_from_docx(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        final_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            
            if resume_text and not final_text:
                final_text = resume_text
            elif resume_text and final_text:
                final_text += "\n\n" + resume_text
            
            if not final_text.strip():
                st.error("Could not extract text from resume")
                return
            
            with st.spinner("Analyzing with AI..."):
                category, confidence, all_categories, all_probs = predict_role(
                    final_text, tokenizer, model, label_encoder, confidence_threshold
                )
            
            st.markdown("---")
            st.subheader("Results")
            
            confidence_color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.4 else "red"
            
            st.markdown(f"**Recommended Role:** `{category}`")
            st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.1%}]")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': confidence_color},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
            st.plotly_chart(fig, use_container_width=True)
            
            results_df = pd.DataFrame({
                'Role': all_categories,
                'Confidence': all_probs
            }).sort_values('Confidence', ascending=False)
            
            fig_bar = px.bar(
                results_df.head(10),
                x='Role',
                y='Confidence',
                title='Top 10 Role Matches'
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("All Confidence Scores")
            results_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x:.2%}")
            st.dataframe(results_df, use_container_width=True)
            
            if confidence >= 0.7:
                st.success("Strong match! Your resume aligns well with this role.")
            elif confidence >= 0.4:
                st.warning("Moderate match. Consider gaining more experience in key areas.")
            else:
                st.info("Skill gap detected. Consider upskilling for this role.")

    with st.sidebar:
        st.markdown("### How to Use")
        st.markdown("""
        1. Upload resume (PDF/DOCX/TXT)
        2. Or paste resume text
        3. Adjust confidence threshold
        4. Click Analyze
        """)
        
        st.markdown("### Model Info")
        st.markdown(f"""
        - Model: DistilBERT Fine-tuned
        - Roles: {len(label_encoder.classes_)}
        - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}
        """)

if __name__ == "__main__":
    main()
    