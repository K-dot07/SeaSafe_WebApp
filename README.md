# SeaSafe Web App (Streamlit)

This is a free, deployable web app for **SeaSafe: Predicting Passenger Survival in Maritime Disasters Using Machine Learning**.

## 🚀 Run Locally (Free)

1. **Download the repository files** (or the ZIP from ChatGPT).
2. Put your dataset at `data/train.csv`. (You can also upload it in the app UI.)
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # Mac/Linux
   ```
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL shown (e.g., http://localhost:8501).

## ☁️ Deploy Free on Streamlit Community Cloud

1. Push these files to a **public GitHub repo**.
2. Go to **share.streamlit.io** and sign in.
3. Click **New app** → select your repo, branch (main), and file **app.py**.
4. Click **Deploy**. (Optional: keep `data/train.csv` in the repo or upload a CSV via the app UI.)

## 📦 Files

- `app.py` – Streamlit app with training, prediction UI, and charts.
- `requirements.txt` – Python dependencies.
- `data/train.csv` – (Add this yourself) Titanic or SeaSafe-format dataset. If missing, the app uses a small public sample.