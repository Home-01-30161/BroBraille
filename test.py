from paddleocr import PaddleOCR
import paddle

print("Paddle version:", paddle.__version__)
print("GPU available:", paddle.is_compiled_with_cuda())

ocr = PaddleOCR(
    lang="th",
    use_textline_orientation=True  
)
result = ocr.ocr("test.jpg")
print(result)
