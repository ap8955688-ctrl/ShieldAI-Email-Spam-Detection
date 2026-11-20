# ShieldAI — Email & SMS Spam Detection

A polished Streamlit dashboard for a supervised text-classification demonstration. It uses TF-IDF features and a Multinomial Naive Bayes classifier to label a message as `spam` or `ham`.

## Run locally (Windows / VS Code)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## Features
- Dark, responsive Streamlit dashboard
- Message scanner with sample messages
- Spam/ham result and model-estimated confidence
- Recent scan history and CSV export
- Dataset composition chart
- Confusion matrix and classification report
- Project information and workflow explanation

## Dataset
The included `data/sms_spam.csv` is a small illustrative starter dataset (60 examples). It is provided so the application runs immediately. Its performance metrics are not meaningful evidence of real-world accuracy. Replace it with a larger, properly labeled public dataset before making performance claims.

CSV columns:
- `label`: `ham` or `spam`
- `message`: message text

## Model workflow
1. Load and clean the CSV with Pandas.
2. Split into train and test sets.
3. Convert text to TF-IDF vectors.
4. Train Multinomial Naive Bayes.
5. Evaluate on held-out examples.
6. Predict the label of user-entered text.

## Limitations
This is an academic prototype, not an email gateway or phishing protection product. It does not inspect sender reputation, URLs, attachments, headers or email authentication. A high confidence score is not a guarantee that a message is safe.
