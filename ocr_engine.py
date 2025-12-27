from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(
    lang="th",
    use_angle_cls=False
)

def run_ocr(img):
    result = ocr.ocr(img)
    texts = []
    if not result:
        return texts
    for page in result:
        for item in page:
            # item ต้องเป็น list/tuple และมีอย่างน้อย 2 ตัว
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            info = item[1]
            # กรณีปกติ (text, confidence)
            if isinstance(info, (list, tuple)):
                if len(info) >= 1:
                    text = info[0]
                else:
                    continue
            # กรณีผิดรูปแบบ เป็น string
            elif isinstance(info, str):
                text = info
            else:
                continue
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts
