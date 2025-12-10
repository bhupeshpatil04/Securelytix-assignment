# OCR_PII_Pipeline_BhupeshPatil

Contents:
- ocr_pii_pipeline.py  -- main pipeline script
- OCR_PII_Pipeline.ipynb -- Jupyter notebook skeleton
- requirements.txt -- python dependencies
- output_results_template.txt -- template for results
- images: page_35.jpg, page_30.jpg, page_14.jpg (the samples you uploaded)

## How to run (locally)
1. Create a virtual environment (recommended):
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows PowerShell

2. Install dependencies:
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

3. Run the script on an image:
   python ocr_pii_pipeline.py page_35.jpg --redact

4. Outputs:
   - page_35_ocr.txt : OCR text + cleaned text + detected PII
   - page_35_redacted.jpg : redacted image (if --redact used)

## Notes / Limitations
- EasyOCR may produce imperfect results on cursive or messy handwriting.
- You can tune preprocessing in the script (thresholding, denoise, deskew) for better accuracy.

