import fitz
import numpy as np
import cv2

def load_pdf(path):
    doc = fitz.open(path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images
