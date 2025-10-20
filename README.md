# ML_Veridia
# 📄 Resume Classification using DistilBERT

### Internship @ **Veridia.io**

---

## 🎯 Objective
Fine-tune a model that classifies resumes into different categories, similar to an ATS (Applicant Tracking System).

---

## 💡 Project Overview
This project involves fine-tuning a **DistilBERT-based resume classifier** to automate job category prediction from raw resume text.  
The model is designed for **real-time classification** and **deployed on AWS** for scalable inference.

---

## 🧭 Approach Followed

### 1. 🧩 Exploratory Data Analysis (EDA)
- Inspected dataset shape, missing values, and class distribution to identify imbalances across resume categories.  
- Examined average **word** and **token counts** per resume to set the optimal `max_length = 512` for tokenization.  
- Visualized TF-IDF features to explore relationships between word frequency patterns and category clusters.

### 2. 🧹 Data Preprocessing
- Converted all text to lowercase.  
- Removed special characters, URLs, digits, and excessive whitespace.  
- Eliminated stopwords to retain only meaningful tokens.  
- Applied **advanced lemmatization** using `WordNetLemmatizer` for consistent base-word representation (e.g., “running” → “run”).

### 3. 🤖 Model Selection
- Selected **DistilBERT**, a lightweight variant of BERT, balancing efficiency and contextual understanding.

### 4. 🔠 Tokenization
- Utilized the **DistilBERT tokenizer** to convert text into token IDs.  
- Applied **padding** and **truncation** (`max_length = 512`) to ensure uniform sequence lengths.

### 5. 🏋️ Training
- Used a **Weighted Trainer** to handle class imbalance.  
- Fine-tuned for **9 epochs** with `learning_rate = 3e-5` using the **AdamW optimizer**.  
- Monitored **loss**, **accuracy**, **F1-score**, **precision**, and **recall** for both training and validation.  

### 6. 📊 Evaluation
- Achieved **F1-score > 0.85** with minimal overfitting and smooth learning curves.  
- Analyzed **classification metrics** and **confusion matrix** for per-class performance.  
- Applied **softmax temperature scaling (T = 0.2)** to enhance prediction confidence calibration.

### 7. 💾 Model Saving
- Exported:
  - Fine-tuned **DistilBERT model**
  - **Tokenizer**
  - **Label Encoder (`label_encoder.pkl`)**
- Ensured full reproducibility for future deployment.

### 8. ☁️ Deployment
- Integrated into a **Streamlit web app** for user-friendly resume classification.  
- Deployed on **AWS** (e.g., SageMaker / Lambda) for scalable, real-time inference.  
- Designed for integration with ATS or HR applications.

---

## 🧰 Tech Stack / Frameworks Used

| Category | Tools & Libraries |
|-----------|-------------------|
| **Programming** | Python |
| **Deep Learning & NLP** | PyTorch, Transformers (Hugging Face), DistilBERT, BERT |
| **Text Processing** | NLTK (lemmatization, tokenization), Stopword Removal, Advanced Cleaning |
| **Data Handling & Analysis** | Pandas, NumPy, Matplotlib, Seaborn |
| **Machine Learning Utilities** | Scikit-learn (LabelEncoder, Metrics) |
| **Deployment & Web App** | Streamlit, AWS (SageMaker / Lambda) |
| **Version Control & Environment** | Git, Jupyter Notebook |

---

## 🚀 Outcome
Successfully developed and deployed a **fine-tuned DistilBERT resume classifier** capable of accurately predicting job categories from resumes, supporting scalable integration with real-world hiring systems.

---

## 👨‍💻 Author
**Nannuri Sai Kamal**  
*Intern – Veridia.io*  
B.Tech Mechatronics | IIT Bhilai  
