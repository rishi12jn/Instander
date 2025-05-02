# Instander
A machine learning system that detects hate speech in text using a fine-tuned BERT model, enhanced with mixed precision training and hyperparameter optimization via the Flower Pollination Algorithm (FPA).

---

## 📌 Features

- ✅ **BERT-based Transformer model** for contextual text classification
- ✅ **Multi-class hate speech detection** (e.g., hate speech, offensive language, neutral)
- ✅ **Mixed precision training** for faster training with less memory
- ✅ **Custom PyTorch dataset class** for efficient batching
- ✅ **Hyperparameter optimization** using Flower Pollination Algorithm (FPA)
- ✅ **Accuracy evaluation** on test data
- ✅ **Model saving** for inference/deployment

---

## 🧠 Model Architecture

- **Base Model**: BERT (`bert-base-uncased`)
- **Classification Head**: Linear layer with softmax over 3 classes
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Evaluation Metric**: Accuracy Score

---

## 🗂️ Dataset

- **Input**: `cleaned_tweet` column from a preprocessed CSV dataset
- **Output Labels**: Encoded into integers for classification (`label`)
- **Preprocessing**: Cleaning, tokenization, and label encoding
- **Train-Test Split**: 80/20

---

## 🔧 Hyperparameters

The following parameters are optimized using FPA:
- **Learning Rate**
- **Batch Size**
- **Weight Decay**

---

## 🌸 Flower Pollination Algorithm (FPA)

FPA is a nature-inspired metaheuristic algorithm modeled after the pollination process of flowering plants. This project uses FPA to search the hyperparameter space efficiently through:

- **Global Pollination (Levy Flights)** for exploration
- **Local Pollination (Solution Crossover)** for exploitation

---

## 🖥️ Requirements

Install dependencies using pip:

```bash
pip install torch transformers pandas scikit-learn tqdm
```

Make sure a CUDA-compatible GPU is available for faster training.

---

## 🚀 How to Run

1. **Preprocess your dataset** and save it as `preprocessed_dataset.csv`.

2. **Run the training script**:
   ```bash
   python bert_fpa_hate_speech.py
   ```

3. The script will:
   - Optimize hyperparameters using FPA
   - Train the final BERT model
   - Save the trained model as `optimized_bert_hate_speech.pth`

---

## 📊 Output

- `optimized_bert_hate_speech.pth`: Trained model weights
- Console logs with:
  - Best hyperparameters found
  - Training progress
  - Accuracy scores

---

## 📈 Example Accuracy

| Metric      | Score |
|-------------|-------|
| Accuracy    | ~96.48%  |
| Classes     | Hate Speech, Offensive, Neutral |

*(Scores may vary based on the dataset and system hardware.)*

---

## 🤖 Technologies Used

- **Python 3.9+**
- **PyTorch**
- **Transformers (Hugging Face)**
- **NumPy, Pandas, Scikit-learn**
- **Mixed Precision (AMP)**
- **Flower Pollination Algorithm (custom implementation)**

---

## 📎 File Structure

```
.
├── bert_fpa_hate_speech.py        # Main script
├── preprocessed_dataset.csv       # Input dataset (not included)
├── optimized_bert_hate_speech.pth # Trained model (output)
└── README.md                      # Project documentation
```

---

## 🧪 Future Improvements

- Add **LIME/SHAP explainability**
- Deploy as a **Flask or FastAPI** web app
- Enable **real-time tweet monitoring** using Twitter API
- Extend to **multi-lingual hate speech detection**

---

## 🧑‍💻 Author

Developed by Rishi Raj Jain


