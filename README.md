# 🩺 AlpaCare Medical Assistant

AlpaCare Medical Assistant is an **AI-powered educational medical chatbot** designed to provide informative and safe responses to healthcare-related queries.  
The project fine-tunes the **Llama-2 model** using **Parameter-Efficient Fine-Tuning (PEFT)** and **LoRA** to enhance medical knowledge while maintaining ethical boundaries.

---

## 🌐 Project Overview

The goal of this project is to create a **safe and educational conversational assistant** that can explain medical conditions, symptoms, and treatments — **without offering real medical advice or diagnosis**.  
It is built for **academic and research purposes**, showcasing how advanced LLMs can be adapted responsibly for the healthcare domain.

### Key Features
- Fine-tuned **Llama-2** model with **PEFT + LoRA**
- Preprocessed dataset focused on **medical education**
- **Content safety filters** and **automatic disclaimers**
- Modular scripts for **training**, **inference**, and **data handling**

---

## 🧠 Architecture & Approach

### 🔹 Model Architecture
The system is based on:
- **Base Model**: Llama-2 (7B/13B)
- **Fine-tuning Technique**: LoRA (Low-Rank Adaptation) via PEFT
- **Training Framework**: Hugging Face Transformers + TRL (Reinforcement Learning with Human Feedback support)

### 🔹 System Workflow
1. **Data Preparation** – Medical educational dataset is cleaned and formatted into instruction-based samples.  
2. **Model Fine-Tuning** – Using LoRA adapters for efficient and low-resource training.  
3. **Evaluation & Inference** – The model generates responses with embedded safety filters.  
4. **Result Storage** – Outputs and logs are stored in `/results` for later analysis.

```
+----------------------+
|  Medical Dataset     |
+----------+-----------+
           |
           v
+----------+-----------+
|  Preprocessing       |
|  (data_loader.py)    |
+----------+-----------+
           |
           v
+----------+-----------+
|  Fine-Tuning Model   |
|  (train_model.py)    |
+----------+-----------+
           |
           v
+----------+-----------+
|  Inference Engine    |
|  (inference.py)      |
+----------+-----------+
           |
           v
+----------+-----------+
|  Safe Educational    |
|  Medical Responses   |
+----------------------+
```

---

## ⚙️ How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/akshatamandhare/AlpaCare-Medical-Assistant.git
cd AlpaCare-Medical-Assistant
```

### Step 2: Setup Environment
```bash
python -m venv alpacare_env
source alpacare_env/bin/activate       # On Windows: alpacare_env\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Login to HuggingFace
```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 4: Run the Model
```bash
# Quick training & inference
python run.py quick

# Full training
python run.py train

# Inference only (with pretrained adapter)
python run.py inference
```

---

## 📦 Dependencies

| Library | Version (min) | Purpose |
|----------|----------------|---------|
| torch | 2.0.0 | Deep learning framework |
| transformers | 4.35.0 | Model loading and tokenization |
| peft | 0.6.0 | LoRA-based fine-tuning |
| datasets | 2.14.0 | Dataset handling |
| accelerate | 0.20.0 | Multi-GPU training |
| bitsandbytes | 0.41.0 | Memory-efficient optimization |
| trl | 0.7.0 | RLHF fine-tuning support |
| jupyter | Latest | Notebook support |

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Information

### 📁 Source
- The dataset can be a **custom educational medical dataset** or obtained from public sources like:
  - [Hugging Face Medical Datasets](https://huggingface.co/datasets)
  - [PubMedQA](https://pubmedqa.github.io/)
  - [MedQA](https://github.com/jind11/MedQA)

### 🧩 Format
Each record should be in instruction format:
```json
{
  "instruction": "Explain what hypertension is.",
  "input": "",
  "output": "Hypertension is a medical condition where blood pressure levels remain higher than normal..."
}
```

### 🔍 Preprocessing
Use `data_loader.py` to clean, tokenize, and prepare the dataset before training.

---

## 🎯 Expected Outputs

After successful training and inference:
- The model should **generate safe and educational responses** to medical queries.
- All outputs include a **medical disclaimer**.
- Example response:

**Input:**  
> What are the symptoms of diabetes?

**Output:**  
> Diabetes symptoms may include increased thirst, frequent urination, fatigue, and blurred vision.  
> *Disclaimer: This information is for educational purposes only and should not be used for diagnosis or treatment.*

---

## 🧰 Troubleshooting

**CUDA Out of Memory**
```python
config['training']['per_device_train_batch_size'] = 1
config['training']['gradient_accumulation_steps'] = 8
```

**Slow Training**
```python
config['subset_size'] = 1000
config['training']['max_steps'] = 100
```

**HuggingFace Login Issues**
```bash
huggingface-cli logout
huggingface-cli login
```

---

## ⚠️ Ethical & Safety Note

> This model is intended **strictly for educational purposes**.  
> It must **not** be used for medical diagnosis, treatment, or clinical decisions.

Always consult a certified medical professional for health concerns.

---

## 🧩 Next Steps

1. Fine-tune with additional data  
2. Evaluate with domain experts  
3. Enhance dataset for multilingual support  
4. Deploy safely using APIs or web interfaces  

---

## 👨‍💻 Author

**Akshata Mandhare**  
GitHub: [akshatamandhare](https://github.com/akshatamandhare)  

---

<div align="center">
  
**⚡ Built with ❤️ for better healthcare accessibility ⚡**

[🌟 Star this repo](https://github.com/akshatamandhare/AlpaCare-Medical-Assistant) • [🍴 Fork](https://github.com/akshatamandhare/AlpaCare-Medical-Assistant/fork) • [📖 Documentation](https://github.com/akshatamandhare/AlpaCare-Medical-Assistant/wiki)

</div>
