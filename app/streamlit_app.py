"""Kinyarwanda tweet sentiment: Streamlit app around a fine-tuned AfroXLMR-large classifier."""
import os
import re

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128


def get_setting(name: str, default: str) -> str:
    """Read a setting from Streamlit secrets, then the environment, then the default."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


# Hub id of the fine-tuned model (uploaded by notebook 03b). If the repo is private, add HF_TOKEN as a secret.
MODEL_ID = get_setting("MODEL_ID", "Gkea/kinyarwanda-sentiment-afroxlmr-large")
HF_TOKEN = get_setting("HF_TOKEN", "") or None
# Set LOW_MEMORY=1 to load the weights in bfloat16 (about half the RAM). Scores can change slightly,
# so re-evaluate the model in this mode before quoting its numbers.
LOW_MEMORY = get_setting("LOW_MEMORY", "0") == "1"

st.set_page_config(page_title="Kinyarwanda Sentiment", page_icon="💬")


@st.cache_resource(show_spinner="Loading the model (the first start can take a few minutes)...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    kwargs = {"torch_dtype": torch.bfloat16} if LOW_MEMORY else {}
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=HF_TOKEN, **kwargs)
    model = model.eval() if LOW_MEMORY else model.float().eval()
    return tok, model


def clean(text: str) -> str:
    """Drop @mentions and collapse whitespace; case, emojis, hashtags and links are kept.

    In the training data every mention is already anonymised to the token "@user", and the training
    cleaner removed exactly that token. Here any @handle is removed, which gives the same result on
    dataset-style text and also handles real usernames that a person might paste in.
    """
    return re.sub(r"\s+", " ", re.sub(r"@\w+", " ", text)).strip()


def classify(text: str):
    tok, model = load_model()
    cleaned = clean(text)
    enc = tok(cleaned, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.inference_mode():
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[0]
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return {label: float(p) for label, p in zip(labels, probs)}, cleaned


# Written for this demo, not taken from the dataset
EXAMPLES = {
    "Positive (demo text)": "Uyu munsi ndishimye cyane, ibintu byose byagenze neza.",
    "Negative (demo text)": "Ibi ni bibi cyane, ndarambiwe iyi serivisi mbi.",
    "Neutral (demo text)": "Inama izaba ejo saa tatu mu cyumba cy'inama.",
}
# Tweets from the AfriSenti test set that the final model gets wrong (checked against test_predictions_final.csv).
FAILURES = {
    "Hard: irony (gold negative, model says positive)": "Ibyiza byo kunywa itabi: 1. Upfa udashaje kuko upfa hakiri kare. 2.Ntagisambo cyakwiba iwawe kuko urara ukorora.",
    "Hard: short reply (gold positive, model says neutral)": "Harya nangahe ngo nyohereze",
    "Hard: neutral read as negative": "Koko nawe ngo wigeze kumara iminsi muri coma",
}
ALL_EXAMPLES = {**EXAMPLES, **FAILURES}

st.title("Kinyarwanda tweet sentiment")
st.write("Type a tweet in Kinyarwanda to see whether the model reads it as **positive**, **neutral** or **negative**.")

if "tweet" not in st.session_state:
    st.session_state["tweet"] = ""

choice = st.selectbox("Or pick an example", ["(none)"] + list(ALL_EXAMPLES))
if choice != "(none)" and st.session_state.get("last_choice") != choice:
    st.session_state["tweet"] = ALL_EXAMPLES[choice]
st.session_state["last_choice"] = choice

text = st.text_area("Tweet", key="tweet", height=120, placeholder="Andika hano...")

if st.button("Classify", type="primary"):
    if not clean(text):
        st.warning("Enter a Kinyarwanda tweet first.")
    else:
        scores, cleaned = classify(text)
        top = max(scores, key=scores.get)
        st.subheader(f"Prediction: {top} ({scores[top]:.0%})")
        st.bar_chart(scores)
        st.caption(f"Text sent to the model: {cleaned}")

st.markdown(
    "**About the model.** [AfroXLMR-large](https://huggingface.co/Davlan/afro-xlmr-large) "
    "(Alabi et al., 2022) fine-tuned on the Kinyarwanda tweets of "
    "[AfriSenti](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment) "
    "(Muhammad et al., 2023), about 3,300 training tweets. Macro-F1 on the AfriSenti test set is 0.697 "
    "(accuracy 0.693).\n\n"
    "**Limitations.** Neutral is the hardest class (F1 0.65). Misclassified tweets include irony, jokes and announcements with "
    "positive wording that annotators labelled neutral, and the scores are the model's confidence, not a guarantee. The model was trained on tweets "
    "and may not transfer to other kinds of text. The model is large and runs on a CPU, "
    "so the first request can take a while."
)
