# ocr_pii_pipeline.py
# OCR + PII Extraction Pipeline for Handwritten Documents
# Requires: opencv-python, easyocr, spacy, numpy
# Install: pip install -r requirements.txt
# Download spacy model: python -m spacy download en_core_web_sm

import cv2
import numpy as np
import easyocr
import spacy
import re
import sys
from pathlib import Path

reader = easyocr.Reader(['en'])  # may take some time on first run
nlp = spacy.load("en_core_web_sm")

def preprocess_image_cv(img):
    # img: BGR image (numpy array)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive/otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(thresh, 3)
    # deskew
    coords = np.column_stack(np.where(denoised > 0))
    if coords.size == 0:
        return denoised
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = denoised.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def ocr_image(img):
    # Accept either file path or numpy image
    if isinstance(img, str) or isinstance(img, Path):
        img = str(img)
        results = reader.readtext(img)
    else:
        results = reader.readtext(img)
    text = " ".join([res[1] for res in results])
    return results, text

def clean_text(text):
    text = text.replace("\\n", " ").strip()
    text = re.sub(r'[^A-Za-z0-9\\s@._:-]', ' ', text)
    text = re.sub(r'\\s+', ' ', text)
    return text

def extract_pii(text):
    pii = {}
    pii['emails'] = re.findall(r'\\b[\\w.-]+?@\\w+?\\.\\w+?\\b', text)
    pii['phones'] = re.findall(r'\\b\\d{10}\\b', text)
    doc = nlp(text)
    pii['names'] = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return pii

def redact_pii_on_image(image_path, detected_terms, output_path):
    img = cv2.imread(image_path)
    results = reader.readtext(img)
    for (bbox, txt, prob) in results:
        for term in detected_terms:
            if term.strip() == "": continue
            if term.lower() in txt.lower():
                pts = np.array(bbox, np.int32).reshape((-1,1,2))
                cv2.fillPoly(img, [pts], (0,0,0))
    cv2.imwrite(output_path, img)

def process_file(image_path, redact=False, out_dir="."):
    img = cv2.imread(image_path)
    prep = preprocess_image_cv(img)
    results, raw_text = ocr_image(prep)
    cleaned = clean_text(raw_text)
    pii = extract_pii(cleaned)
    basename = Path(image_path).stem
    out_text_file = Path(out_dir) / f"{basename}_ocr.txt"
    with open(out_text_file, "w", encoding="utf-8") as f:
        f.write("EXTRACTED TEXT:\\n")
        f.write(raw_text + "\\n\\n")
        f.write("CLEANED TEXT:\\n")
        f.write(cleaned + "\\n\\n")
        f.write("DETECTED PII:\\n")
        f.write(str(pii) + "\\n")
    if redact:
        detected_terms = pii.get('emails', []) + pii.get('phones', []) + pii.get('names', [])
        out_redact = Path(out_dir) / f"{basename}_redacted.jpg"
        redact_pii_on_image(image_path, detected_terms, str(out_redact))
    return raw_text, cleaned, pii

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python ocr_pii_pipeline.py <image_path> [--redact]")
        sys.exit(1)
    img_path = sys.argv[1]
    redact_flag = '--redact' in sys.argv
    process_file(img_path, redact=redact_flag, out_dir='.')
    print("Done. Check generated _ocr.txt and _redacted.jpg (if requested).")
