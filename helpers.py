import fitz

def extract_text_from_pdf(pdf_path):
    text = ""
    # Відкриваємо документ
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def handle_file(file):
    if file is None:
        return ""

    text = extract_text_from_pdf(file.name)
    return text