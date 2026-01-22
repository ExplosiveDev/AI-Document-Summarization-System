import gradio as gr
from LLM import update_counter, process_text
from helpers import handle_file


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# BART Summarization ")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF document", file_types=[".pdf"])

            input_text = gr.Textbox(label="Input text", placeholder="Insert news or paper", lines=15)

            file_input.change(fn=handle_file, inputs=file_input, outputs=input_text)

            char_count = gr.Markdown("Symbols: 0, Tokens: 0")

            input_text.change(fn=update_counter, inputs=input_text, outputs=char_count)

            with gr.Accordion("Options", open=False):
                min_length = gr.Slider(10, 50, value=20, step=5, label="Min. length")
                max_length = gr.Slider(60, 200, value=100, step=10, label="Max. length")

                temperature = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="Temperature (Creativity)")
                top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p (Nucleus Sampling)")

                chunks_on = gr.Checkbox(value=True, label="Processing long texts")
            btn = gr.Button("Summarized text", variant="primary")


        with gr.Column():
            output_text = gr.Textbox(label="Summarize", lines=15)

    btn.click(
        fn=process_text,
        inputs=[input_text, min_length, max_length, temperature, top_p, chunks_on],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()