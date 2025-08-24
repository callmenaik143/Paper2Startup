import gradio as gr
import os
import PyPDF2
from groq import Groq

# ============================================
# ✅ Config: API key + Model
# ============================================
# IMPORTANT: Do NOT hardcode your key here when deploying.
# Instead, add it in Hugging Face "Secrets" as GROQ_API_KEY.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

# ============================================
# ✅ Helpers: Read PDF + Chunk
# ============================================
def read_pdf(file_path):
    try:
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def chunk_text(text, max_words=1000):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ============================================
# ✅ Chat with Groq
# ============================================
def groq_chat(prompt):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI that explains research in business-friendly language."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# ============================================
# ✅ Main Logic
# ============================================
def analyze_paper(pdf):
    if pdf is not None:
        text = read_pdf(pdf)
    else:
        return "No file uploaded.", "", "", ""

    if not text or text.startswith("Error"):
        return "No text found in paper.", "", "", ""

    chunks = chunk_text(text, max_words=1000)

    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        summary = groq_chat(f"Summarize this research paper section (Part {i}) in simple plain language:\n\n{chunk}")
        chunk_summaries.append(summary)

    combined_summary = "\n".join(chunk_summaries)

    compressed_summary = groq_chat(
        "Combine and compress these summaries into a single plain-language overview, "
        "no longer than 500 words:\n\n" + combined_summary
    )

    use_cases = groq_chat("Suggest practical startup use cases for this research:\n\n" + compressed_summary)
    pitch = groq_chat("Write a draft pitch deck outline (Problem, Solution, Market, Product, Team, Why Now):\n\n" + compressed_summary)
    monet = groq_chat("Suggest possible monetization models for a startup based on this research:\n\n" + compressed_summary)

    return compressed_summary, use_cases, pitch, monet

# ============================================
# ✅ Gradio UI
# ============================================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #2E86C1;'>📄 Paper2Startup</h1>
        <p style='text-align: center; font-size:16px;'>Upload your research paper and convert it into startup opportunities 🚀</p>
        """,
    )

    with gr.Row():
        pdf_input = gr.File(label="📂 Upload your Research Paper (PDF)", file_types=[".pdf"], type="filepath")

    analyze_btn = gr.Button("🚀 Analyze Paper")

    with gr.Tabs():
        with gr.Tab("📌 Summary"):
            summary_out = gr.Textbox(label="Plain-language Summary", lines=12, interactive=False)
        with gr.Tab("💡 Use Cases"):
            usecases_out = gr.Textbox(label="Startup Use Cases", lines=12, interactive=False)
        with gr.Tab("📊 Pitch Deck"):
            pitch_out = gr.Textbox(label="Pitch Deck Draft", lines=12, interactive=False)
        with gr.Tab("💰 Monetization"):
            monet_out = gr.Textbox(label="Monetization Models", lines=12, interactive=False)

    analyze_btn.click(
        analyze_paper,
        inputs=[pdf_input],
        outputs=[summary_out, usecases_out, pitch_out, monet_out]
    )

# ============================================
# ✅ Launch
# ============================================
demo.launch()
