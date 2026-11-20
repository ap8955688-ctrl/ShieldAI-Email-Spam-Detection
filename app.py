from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "sms_spam.csv"

st.set_page_config(
    page_title="ShieldAI | Message Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #edf2ff; }
[data-testid="stHeader"] { background: rgba(11,16,32,0); }
[data-testid="stSidebar"] { background: #111a2e; border-right: 1px solid #25314b; }
[data-testid="stSidebar"] * { color: #eaf0ff !important; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #f5f7ff !important; }
p, label, .stMarkdown { color: #c5cee2; }
.hero {
  background: linear-gradient(120deg, #17294b 0%, #18213e 55%, #25204a 100%);
  border: 1px solid #354568; border-radius: 22px; padding: 28px 30px;
  margin-bottom: 22px; box-shadow: 0 18px 50px rgba(0,0,0,.18);
}
.hero-kicker { color:#92b8ff; font-size:12px; letter-spacing:2px; font-weight:700; text-transform:uppercase; }
.hero-title { font-family:'Manrope'; font-size:34px; line-height:1.15; font-weight:800; color:#fff; margin:8px 0; }
.hero-sub { color:#bdc9e2; font-size:15px; max-width:760px; }
.metric-card {
 background: linear-gradient(145deg,#151f35,#121a2d); border:1px solid #2b3956;
 border-radius:16px; padding:18px 20px; min-height:112px;
}
.metric-label { color:#8fa1c2; font-size:12px; text-transform:uppercase; letter-spacing:1px; font-weight:700; }
.metric-value { color:#f6f8ff; font-family:'Manrope'; font-size:27px; font-weight:800; margin-top:6px; }
.metric-note { color:#8fa1c2; font-size:12px; margin-top:2px; }
.panel {
 background:#121b30; border:1px solid #293854; border-radius:18px;
 padding:22px; margin-bottom:16px;
}
.section-eyebrow { color:#8daeff; font-size:11px; font-weight:800; letter-spacing:1.6px; text-transform:uppercase; }
.result-spam {
 background:linear-gradient(120deg,rgba(239,68,68,.18),rgba(127,29,29,.12));
 border:1px solid rgba(248,113,113,.55); border-radius:18px; padding:22px;
}
.result-ham {
 background:linear-gradient(120deg,rgba(16,185,129,.16),rgba(6,95,70,.10));
 border:1px solid rgba(52,211,153,.5); border-radius:18px; padding:22px;
}
.result-title { font-family:'Manrope'; font-size:25px; font-weight:800; color:#fff; }
.small-muted { color:#93a3c1; font-size:12px; }
div.stButton > button {
 border-radius:10px; border:1px solid #40547b; background:#24385e; color:white;
 font-weight:700; min-height:42px;
}
div.stButton > button:hover { border-color:#8bb2ff; color:#fff; background:#2b4777; }
.stTextArea textarea, .stTextInput input {
 background:#0e1729 !important; color:#edf2ff !important; border:1px solid #354563 !important;
 border-radius:12px !important;
}
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#151f35; border-radius:9px 9px 0 0; color:#c5cee2; padding:10px 18px; }
.stTabs [aria-selected="true"] { background:#263d67 !important; color:white !important; }
hr { border-color:#293854; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])]
    return df

@st.cache_resource
def train_model():
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=5000, sublinear_tf=True
        )),
        ("model", MultinomialNB(alpha=0.5))
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    return pipeline, df, y_test, pred, accuracy_score(y_test, pred)

model, dataset, y_test, y_pred, accuracy = train_model()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🛡️ ShieldAI")
    st.markdown("<div class='small-muted'>MESSAGE THREAT INTELLIGENCE</div>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", ["Message Scanner", "Analytics", "Project Info"], label_visibility="visible")
    st.divider()
    st.markdown("**MODEL STATUS**")
    st.success("Model trained and ready")
    st.caption(f"Training dataset: {len(dataset)} sample messages")
    st.caption("Classifier: Multinomial Naive Bayes")
    st.caption("Features: TF-IDF word + phrase patterns")
    st.divider()
    st.markdown("<div class='small-muted'>ACADEMIC DEMONSTRATION • PYTHON + MACHINE LEARNING</div>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">AI-powered message protection</div>
  <div class="hero-title">Detect suspicious messages<br>before they reach you.</div>
  <div class="hero-sub">Analyze message text with a machine-learning classifier. ShieldAI identifies patterns commonly found in spam and separates them from ordinary messages.</div>
</div>
""", unsafe_allow_html=True)

# Metric strip
a, b, c, d = st.columns(4)
metrics = [
    (a, "MODEL ACCURACY", f"{accuracy:.1%}", "On held-out demo data"),
    (b, "MESSAGES IN DATASET", f"{len(dataset):,}", "Labeled examples"),
    (c, "SPAM EXAMPLES", f"{(dataset.label == 'spam').sum():,}", "Training corpus"),
    (d, "SCANS THIS SESSION", f"{len(st.session_state.history):,}", "Current browser session"),
]
for col, label, value, note in metrics:
    with col:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)

st.write("")

if page == "Message Scanner":
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-eyebrow'>01 / INPUT</div>", unsafe_allow_html=True)
        st.subheader("Scan a message")
        st.write("Paste the content of an email or text message below.")
        examples = {
            "Choose a sample message": "",
            "Suspicious prize message": "Congratulations! You have won a free cash prize. Click this link now to claim your reward.",
            "Suspicious account alert": "URGENT! Your account is blocked. Verify your details immediately to avoid suspension.",
            "Ordinary personal message": "Hey, are we still meeting at the library at 4 pm today?",
            "Ordinary college message": "The lecture starts at 9 am in room 204. Please bring your assignment notes."
        }
        choice = st.selectbox("Try a sample", list(examples.keys()))
        default_text = examples[choice]
        message = st.text_area("MESSAGE CONTENT", value=default_text, height=190,
                               placeholder="Paste or type the message you want to analyze...")
        char_count = len(message)
        st.markdown(f"<div class='small-muted'>{char_count} characters • Text is analyzed locally by this demo model</div>", unsafe_allow_html=True)
        scan = st.button("🔍  Analyze message", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if scan:
            if not message.strip():
                st.warning("Enter a message before starting the scan.")
            else:
                label = model.predict([message])[0]
                probs = model.predict_proba([message])[0]
                classes = list(model.classes_)
                confidence = float(probs[classes.index(label)])
                st.session_state.last_result = {
                    "message": message, "label": label, "confidence": confidence,
                    "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
                }
                st.session_state.history.insert(0, st.session_state.last_result)
                st.session_state.history = st.session_state.history[:20]

        if "last_result" in st.session_state:
            result = st.session_state.last_result
            if result["label"] == "spam":
                st.markdown(f"""
                <div class="result-spam">
                  <div class="section-eyebrow">02 / SCAN RESULT</div>
                  <div class="result-title">🚨 Suspicious message detected</div>
                  <p style="color:#fca5a5;margin-bottom:8px">The model classified this message as <b>SPAM</b>.</p>
                  <div style="font-size:30px;font-weight:800;color:#fff">{result['confidence']:.1%}</div>
                  <div class="small-muted">Model-estimated confidence</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ham">
                  <div class="section-eyebrow">02 / SCAN RESULT</div>
                  <div class="result-title">✅ No spam pattern detected</div>
                  <p style="color:#6ee7b7;margin-bottom:8px">The model classified this message as <b>HAM</b> (not spam).</p>
                  <div style="font-size:30px;font-weight:800;color:#fff">{result['confidence']:.1%}</div>
                  <div class="small-muted">Model-estimated confidence</div>
                </div>""", unsafe_allow_html=True)
            st.caption("Confidence is a model estimate, not a guarantee. A message classified as genuine can still be fraudulent.")
    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-eyebrow'>LIVE ACTIVITY</div>", unsafe_allow_html=True)
        st.subheader("Recent scans")
        if st.session_state.history:
            for item in st.session_state.history[:6]:
                icon = "🚨" if item["label"] == "spam" else "✅"
                short = item["message"].replace("\n", " ")
                short = short[:76] + ("…" if len(short) > 76 else "")
                st.markdown(f"**{icon} {item['label'].upper()}** · {item['confidence']:.0%}")
                st.caption(short)
                st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.info("Your scan activity will appear here after you analyze a message.")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Export scan history", csv_data, "shieldai_scan_history.csv", "text/csv", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Analytics":
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>MODEL INSIGHTS</div>", unsafe_allow_html=True)
    st.subheader("Dataset composition")
    counts = dataset["label"].value_counts().rename_axis("Category").reset_index(name="Messages")
    p1, p2 = st.columns(2)
    with p1:
        fig = px.pie(counts, names="Category", values="Messages", hole=.62,
                     color="Category", color_discrete_map={"ham":"#34d399","spam":"#fb7185"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#dbe5fa", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        st.markdown("**Test-set confusion matrix**")
        cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
        cm_df = pd.DataFrame(cm, index=["Actual genuine", "Actual spam"],
                             columns=["Predicted genuine", "Predicted spam"])
        st.dataframe(cm_df, use_container_width=True)
        st.metric("Held-out test accuracy", f"{accuracy:.1%}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, labels=["ham", "spam"], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
    st.caption("Precision measures how many predicted examples of a class were correct. Recall measures how many actual examples of a class were found.")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-eyebrow'>ABOUT THE PROJECT</div>", unsafe_allow_html=True)
    st.subheader("Email & SMS Spam Detection System")
    st.write("This project demonstrates a supervised machine-learning workflow for classifying message text into two categories: spam and ham.")
    st.markdown("**Technology stack**")
    st.write("- Python — application logic")
    st.write("- Pandas — dataset loading and cleaning")
    st.write("- TF-IDF — converts text into numerical features")
    st.write("- Multinomial Naive Bayes — classification algorithm")
    st.write("- Scikit-learn — model pipeline and evaluation")
    st.write("- Streamlit — interactive dashboard")
    st.markdown("**How the system works**")
    st.write("1. Load labeled messages from a CSV dataset.")
    st.write("2. Split examples into training and testing sets.")
    st.write("3. Convert message text into TF-IDF features.")
    st.write("4. Train a Multinomial Naive Bayes classifier.")
    st.write("5. Predict the class of a new message and display an estimated confidence.")
    st.warning("Academic prototype: the included starter dataset is intentionally small and illustrative. Do not use this app as a real security product or rely on it to decide whether links, attachments or senders are safe.")
    st.markdown("**Future improvements**")
    st.write("- Train on a larger public spam dataset.")
    st.write("- Compare Naive Bayes, Logistic Regression and Linear SVM.")
    st.write("- Add model persistence, user feedback and a larger evaluation set.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><hr><div class='small-muted' style='text-align:center'>SHIELDAI · ACADEMIC MACHINE LEARNING PROJECT · BUILT WITH PYTHON</div>", unsafe_allow_html=True)
