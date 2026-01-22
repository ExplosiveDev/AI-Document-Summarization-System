import torch
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter


device = 0 if torch.cuda.is_available() else -1
model_name = "facebook/bart-large-cnn"

print(f"Завантаження моделі на {'GPU' if device == 0 else 'CPU'}...")


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
)


def recursive_summarize(texts, min_len, max_len, temp, top_p, progress=gr.Progress(), current_logs=[]):
    summaries = []

    current_level_info = f"--- Processing level: {len(texts)} chunks ---"

    for i, txt in enumerate(progress.tqdm(texts, desc=current_level_info)):
        input_txt = "summarize: " + txt
        res = summarizer(
            input_txt,
            max_length=max_len,
            min_length=min_len,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=2.0
        )

        summaries.append(res[0]['summary_text'])
        torch.cuda.empty_cache()

    current_logs.append({
        'level': current_level_info,
        'resume': summaries
    })

    if len(summaries) > 3:
        new_chunks = []
        for i in range(0, len(summaries), 5):
            group = " ".join(summaries[i:i + 5])
            new_chunks.append(group)
        return recursive_summarize(new_chunks, min_len, max_len, temp, top_p, progress, current_logs)

    for log in current_logs:
        print(log['level'])
        for text in log['resume']:
            print(text)

    print("--- Summarized text ---")
    print(summaries)

    return "\n\n".join(summaries)


def process_text(text, min_len, max_len, temp, top_p, chunks_enabled, progress=gr.Progress()):
    if not text.strip():
        return "Please enter text."

    input_main = "summarize: " + text

    if chunks_enabled:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        init_chunks = splitter.split_text(text)
        return recursive_summarize(init_chunks, min_len, max_len, temp, top_p, progress)
        return final_result
    else:
        result = summarizer(input_main, max_length=max_len, min_length=min_len, do_sample=False)
        return result[0]['summary_text']

def update_counter(text):
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return f"Symbols: {len(text)}, Tokens: {tokens}"