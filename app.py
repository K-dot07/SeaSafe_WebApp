import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="SeaSafe - Survival Prediction", page_icon="🌊", layout="wide")

# --------------------------
# Utilities
# --------------------------

@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file=None):
    """
    Load dataset from (priority):
    1) User-uploaded CSV (must look like Titanic/SeaSafe)
    2) Local file data/train.csv (you can add it to your repo)
    3) Public fallback URL (tiny Titanic sample)
    """
    # 1) Uploaded by user
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, "uploaded"
        except Exception as e:
            st.warning(f"Could not read uploaded file: {e}")

    # 2) Local
    local_paths = ["data/train.csv", "train.csv", "data/titanic.csv", "titanic.csv"]
    for p in local_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except Exception as e:
                st.warning(f"Found {p} but could not read it: {e}")

    # 3) Fallback URL
    try:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        return df, "fallback_url"
    except Exception as e:
        st.error("No dataset found. Please upload a CSV similar to the Titanic dataset.")
        return None, None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize different 'SeaSafe' vs 'Titanic' column names to a common schema.
    """
    rename_map = {
        # SeaSafe-style -> Titanic-style
        "PassengerNumber": "PassengerId",
        "TicketClass": "Pclass",
        "Gender": "Sex",
        "PassengerAge": "Age",
        "SiblingsAboard": "SibSp",
        "ParentsChildren": "Parch",
        "TicketFare": "Fare",
        "BoardingPort": "Embarked",
        "CabinNumber": "Cabin",
    }
    cols = {c: rename_map.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df = standardize_columns(df)

    # Keep only columns we need if they exist
    needed = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]
    available = [c for c in needed if c in df.columns]
    df = df[available]

    # Handle missing values
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Drop rows with any remaining NaNs just to be safe
    df = df.dropna()

    # Encode categorical
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2}).astype(int)

    # Features / target
    features = [c for c in ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"] if c in df.columns]
    target = "Survived" if "Survived" in df.columns else None

    return df, features, target

@st.cache_resource(show_spinner=False)
def train_model(df, features, target):
    # If no target (for example, when using test.csv), we just train on rows with target
    if target is None or target not in df.columns:
        st.warning("No 'Survived' column found. The model will be trained only if labels are available.")
        return None, None, None

    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=False)

    return model, acc, report

def predict_one(model, features, form_values):
    row = pd.DataFrame([form_values], columns=features)
    proba = model.predict_proba(row)[0][1]
    pred = int(proba >= 0.5)
    return pred, proba

# --------------------------
# UI
# --------------------------

st.title("🌊 SeaSafe: Predicting Passenger Survival")
st.caption("A simple ML web app that learns from Titanic/SeaSafe data and predicts survival probability.")

with st.expander("📥 Load Dataset", expanded=True):
    up = st.file_uploader("Upload a CSV (Titanic/SeaSafe format). If you skip this, the app will try local files or a public sample.", type=["csv"])
    raw_df, source = load_dataset(up)

if raw_df is None:
    st.stop()

st.success(f"Dataset source: {source}")
st.write("Preview of data:")
st.dataframe(raw_df.head(10), use_container_width=True)

df, features, target = preprocess(raw_df)

if target is None or target not in df.columns:
    st.info("No 'Survived' column detected. You can still explore data, but training needs labels.")
else:
    st.write(f"Using features: `{features}`  |  Target: `{target}`")

# Train model
model, acc, report = train_model(df, features, target)

# Layout
left, right = st.columns([1,1])

with left:
    st.subheader("📊 Model Performance")
    if model is not None:
        st.write(f"Accuracy (hold-out set): **{acc:.2f}**")
        st.text("Classification report:")
        st.text(report)
    else:
        st.warning("Model not trained (labels missing). Upload a labeled CSV with a 'Survived' column.")

with right:
    st.subheader("🧪 Try a Prediction")
    if model is not None:
        # Build input form based on available features
        form = st.form("predict_form")
        form_vals = {}

        if "Pclass" in features:
            form_vals["Pclass"] = form.selectbox("Ticket Class (1=First, 2=Second, 3=Third)", [1,2,3], index=2)
        if "Sex" in features:
            sex_label = form.selectbox("Gender", ["male","female"], index=0)
            form_vals["Sex"] = 0 if sex_label == "male" else 1
        if "Age" in features:
            form_vals["Age"] = form.slider("Age", 0, 80, 30)
        if "SibSp" in features:
            form_vals["SibSp"] = form.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, step=1)
        if "Parch" in features:
            form_vals["Parch"] = form.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0, step=1)
        if "Fare" in features:
            # Use dataset stats for a sensible range
            fare_min = int(df["Fare"].min()) if "Fare" in df.columns else 0
            fare_max = int(df["Fare"].max()) if "Fare" in df.columns else 512
            default_fare = min(max(8, fare_min), fare_max)
            form_vals["Fare"] = form.slider("Ticket Fare", fare_min, fare_max, default_fare)
        if "Embarked" in features:
            emb_label = form.selectbox("Boarding Port", ["C","Q","S"], index=2)
            emb_map = {"C":0,"Q":1,"S":2}
            form_vals["Embarked"] = emb_map[emb_label]

        submitted = form.form_submit_button("Predict Survival")
        if submitted:
            pred, proba = predict_one(model, features, form_vals)
            st.metric("Predicted Survival", "Yes ✅" if pred==1 else "No ❌")
            st.metric("Probability", f"{proba*100:.1f}%")

st.subheader("📈 Insights")

# Gender vs survival
if "Sex" in df.columns and "Survived" in df.columns:
    st.write("Survival by Gender")
    # Convert back to label for plotting
    plot_df = df[["Sex","Survived"]].copy()
    plot_df["Sex"] = plot_df["Sex"].map({0:"male",1:"female"})
    counts = plot_df.groupby(["Sex","Survived"]).size().unstack(fill_value=0)
    fig = plt.figure()
    counts.plot(kind="bar", ax=plt.gca())
    plt.title("Survival by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    st.pyplot(fig)

# Class vs survival
if "Pclass" in df.columns and "Survived" in df.columns:
    st.write("Survival by Ticket Class")
    counts = df.groupby(["Pclass","Survived"]).size().unstack(fill_value=0)
    fig2 = plt.figure()
    counts.plot(kind="bar", ax=plt.gca())
    plt.title("Survival by Ticket Class")
    plt.xlabel("Class (1=First, 3=Third)")
    plt.ylabel("Count")
    st.pyplot(fig2)

# Age distribution by survival
if "Age" in df.columns and "Survived" in df.columns:
    st.write("Age Distribution by Survival")
    survived_ages = df[df["Survived"]==1]["Age"]
    not_survived_ages = df[df["Survived"]==0]["Age"]
    fig3 = plt.figure()
    plt.hist([not_survived_ages, survived_ages], bins=20, label=["No","Yes"], stacked=True)
    plt.title("Age Distribution by Survival")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend(title="Survived")
    st.pyplot(fig3)

# Feature importance
if "Survived" in df.columns and model is not None:
    st.write("Feature Importance (Random Forest)")
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    fig4 = plt.figure()
    importances.plot(kind="barh", ax=plt.gca())
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    st.pyplot(fig4)

st.caption("Tip: Add your own SeaSafe dataset with renamed columns (PassengerAge, Gender, TicketClass, etc.). The app recognizes both Titanic and SeaSafe naming.")