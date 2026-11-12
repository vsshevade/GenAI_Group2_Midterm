"""Streamlit UI for the MS-ADS RAG chatbot.   Run:  streamlit run app.py"""
import os
import streamlit as st

st.set_page_config(page_title="MS-ADS Assistant", page_icon="🎓", layout="wide")

# Key from Streamlit secrets (deployment) or environment (local)
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

import rag_core  # noqa: E402


@st.cache_resource(show_spinner="Loading embedding model and index...")
def load_pipeline():
    return rag_core.RAGPipeline()


rag = load_pipeline()

st.title("🎓 UChicago MS in Applied Data Science Assistant")
st.caption("Answers come only from the official program website. Always verify deadlines, "
           "tuition and policies on the official site.")

with st.sidebar:
    st.header("About")
    st.write("Retrieval-Augmented Generation over datascience.uchicago.edu "
             "(MS in Applied Data Science pages).")
    st.write(f"**Mode:** {'LLM (' + rag_core.LLM_MODEL + ')' if rag.llm else 'Offline (no API key) - shows top passage'}")
    st.write(f"**Indexed chunks:** {rag.collection.count()}")
    st.write("**Safeguards:** PII redaction, prompt-injection filter, out-of-scope refusal, "
             "grounding check, source citations.")
    show_src = st.toggle("Show retrieved sources", value=True)
    st.subheader("Try asking")
    examples = ["What is tuition cost for the program?",
                "What are the deadlines for the in-person program?",
                "What are the minimum TOEFL and IELTS scores?",
                "Is the program STEM/OPT eligible?",
                "How many courses must I complete to earn the degree?",
                "Can you tell me about the capstone project?"]
    clicked = None
    for e in examples:
        if st.button(e, use_container_width=True):
            clicked = e
    if st.button("Clear conversation"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(res):
    if not (show_src and res.get("sources")):
        return
    with st.expander(f"Sources ({len(res['sources'])} retrieved passages)"):
        for i, h in enumerate(res["sources"], 1):
            st.markdown(f"**[{i}] {h['title']} > {h['section']}**  \n"
                        f"similarity {h['score']:.2f} · [{h['url']}]({h['url']}) · scraped {h.get('scraped_on', '')}")
            st.text(h["text"][:600] + ("..." if len(h["text"]) > 600 else ""))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("res"):
            render_sources(m["res"])

question = st.chat_input("Ask about courses, admissions, tuition, faculty, careers...") or clicked
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the program website..."):
            res = rag.answer(question, history=history)
        if res["pii_found"]:
            st.warning("Personal information (" + ", ".join(res["pii_found"]) + ") was removed from your question.")
        st.markdown(res["answer"])
        if res["grounded"] is False:
            st.info("Grounding check flagged unverified details: " + ", ".join(res["unsupported"]))
        render_sources(res)
    st.session_state.messages.append({"role": "user", "content": res["question"]})
    st.session_state.messages.append({"role": "assistant", "content": res["answer"], "res": res})
