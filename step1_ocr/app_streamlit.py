import streamlit as st
import cv2
import numpy as np
import unicodedata
import tempfile
import os

from ocr_engine import run_ocr
from pdf_loader import load_pdf    

st.set_page_config(page_title="Thai OCR (Image / PDF)")
st.title("Thai OCR (Image/PDF)")

uploaded_file = st.file_uploader(
    "Upload as picture(.jpg .jpeg .png) or PDF",
    type=["jpg", "png", "jpeg", "pdf"]
)
if uploaded_file is not None:
    all_text = []
    #PDF
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        images = load_pdf(pdf_path)
        for img in images:
            all_text.extend(run_ocr(img))
    #image
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        all_text = run_ocr(img)

    final_text = "\n".join(all_text)
    final_text = unicodedata.normalize("NFC", final_text)

    st.subheader("OCR Result")
    st.text_area("ข้อความที่ได้", final_text, height=300)

    if final_text.strip():
        os.makedirs("output", exist_ok=True)
        with open("output/text.txt", "w", encoding="utf-8") as f:
            f.write(final_text)

        st.success("log เป็น output/text.txt แล้ว")
    else:
        st.warning("ไม่พบข้อความจาก OCR")
