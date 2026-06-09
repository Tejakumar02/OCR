# Smart Document Extraction Pipeline — Dual OCR Engine (PaddleOCR GPU + Tesseract CPU)

## Overview

Smart Document Extraction Pipeline is a production-ready document processing tool that extracts text, tables, figures, and hyperlinks from PDF documents using parallel processing and dual OCR engine support — PaddleOCR for GPU environments and Tesseract for CPU-only deployments.

This is not an "AI tool" — it is an intelligent document processing pipeline built with computer vision, OCR, and parallel processing. That distinction matters.

## Why Two Versions?

The GPU version (PaddleOCR) is optimized for speed in environments with CUDA-compatible hardware. 
<img width="1799" height="1015" alt="inital page" src="https://github.com/user-attachments/assets/d33ff365-4600-480e-b5bd-3d22e052b159" />
<img width="1829" height="1030" alt="Screenshot 2025-04-20 144720" src="https://github.com/user-attachments/assets/5d46f741-addd-4c22-845c-f1f89e3ae54c" />
<img width="1825" height="622" alt="Screenshot 2025-04-20 144849" src="https://github.com/user-attachments/assets/2d0376d8-fe46-437f-b341-630992ccb4c0" />
<img width="1920" height="1080" alt="Extracted Text" src="https://github.com/user-attachments/assets/bbaffbef-3ca0-42e6-b3e5-e2968f9765b9" />
<img width="1920" height="1080" alt="Extracted tables" src="https://github.com/user-attachments/assets/05f87852-0480-4083-9d8d-7c7663322655" />
<img width="1768" height="937" alt="figure t" src="https://github.com/user-attachments/assets/8a17ed0b-bfa7-4867-85f1-c7d395aca2c9" />
<img width="1768" height="937" alt="figure t" src="https://github.com/user-attachments/assets/f5dab920-c95d-4735-883b-6542be800899" />
<img width="1920" height="1080" alt="links" src="https://github.com/user-attachments/assets/c7641ff3-4a51-47c3-9f27-28567ff7c5f3" />
<img width="1816" height="678" alt="features" src="https://github.com/user-attachments/assets/e89a88a9-9b58-47d2-b5aa-de9cff91cb90" />


The CPU version (Tesseract) is designed for zero-GPU deployment — useful for air-gapped or resource-constrained environments where external dependencies are not allowed.
<img width="1920" height="1080" alt="Screenshot 2025-04-14 121744" src="https://github.com/user-attachments/assets/781f57ee-e43e-4ac5-be05-265e8aa1ae07" />
<img width="1920" height="1080" alt="Screenshot 2025-04-14 121815" src="https://github.com/user-attachments/assets/47047286-c40d-4f08-beb0-013359b1f732" />

---
 TheRemaining Screeshots are similar to PaddleOCR Screenshots, with slight difference.
 Both versions produce identical output formats and follow the same processing pipeline. Choose based on your hardware constraints.
---
## Performance

Tested on an 18MB PDF document with 31 pages:

| Version | Engine | Pages | File Size | Total Extraction Time |
|---|---|---|---|---|
| main.py (GPU) | PaddleOCR | 31 | 18 MB | ~17.76 seconds |
| app.py (CPU) | Tesseract | 31 | 18 MB | ~42.89 seconds |

GPU version is **2.4x faster** than CPU version. Times vary ±2–3 seconds depending on PDF complexity and content density.

---

## Features

**Text Extraction**
- Dual OCR engines: PaddleOCR (GPU-accelerated) and Tesseract (CPU-based)
- Automatic mathematical formula conversion to LaTeX format
- Hyperlink extraction from both OCR text and embedded PDF structure
- XML-safe sanitization of all extracted content — NULL bytes and control characters stripped before export

**Table Extraction**
- Structured table detection using pdfplumber and tabula-py
- Page-wise table organization with CSV export via pandas
- Escape character handling for special characters in table cells

**Figure Detection**
- Automatic figure detection using OpenCV contour analysis with size filtering (min 100x100px)
- Cropped figure export in PNG format

**Export Pipeline**
- Extracted text as Markdown
- Tables as CSV files packaged in ZIP archive
- Figures as PNG files packaged in ZIP archive
- Hyperlinks as TXT files packaged in ZIP archive
- All content combined into a single compiled DOCX output

**Performance**
- Parallel content extraction using ThreadPoolExecutor — text, tables, and figures processed simultaneously per page
- Real-time progress tracking via Streamlit frontend
- Memory-efficient streaming for large files

---

## Processing Pipeline

```
PDF Upload
    ↓
PyMuPDF page rendering
    ↓
ThreadPoolExecutor (parallel per page)
    ├── OCR text (PaddleOCR / Tesseract)
    ├── pdfplumber table extraction
    └── OpenCV figure detection
    ↓
XML sanitization + formula conversion
    ↓
Multi-format export
    ├── Markdown (text)
    ├── CSV ZIP (tables)
    ├── PNG ZIP (figures)
    ├── TXT ZIP (links)
    └── DOCX (all content combined)
```

---

## Installation

**Prerequisites**
- Python 3.8 or higher
- For GPU version: CUDA-compatible hardware with appropriate drivers
- For CPU version: Tesseract OCR installed on system

**GPU Version (PaddleOCR) — main.py**

```bash
git clone https://github.com/Tejakumar02/OCR.git
cd OCR

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install streamlit paddleocr paddlepaddle Pillow PyMuPDF opencv-python numpy pdfplumber tabula-py python-docx pandas
```

**CPU Version (Tesseract) — app.py**

```bash
pip install streamlit pytesseract pillow pymupdf opencv-python numpy tabula-py pdfplumber pandas python-docx

# Install Tesseract OCR engine
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

---

## Usage

```bash
# GPU version
streamlit run main.py

# CPU version
streamlit run app.py
```

Once running:
1. Upload a PDF, image, or text file using the upload area
2. Click "Extract All Content" to process the document
3. Preview extracted text, tables, figures, and links in the interface
4. Download specific content types or everything as a single DOCX file

**Supported input formats:** PDF, PNG, JPG, JPEG, TXT, MD, DOCX

**Supported output formats:** Markdown, CSV, PNG, DOCX, ZIP

---

## Key Components

| Component | Library | Purpose |
|---|---|---|
| Document Parser | PyMuPDF (fitz) | PDF page rendering and embedded link extraction |
| OCR Engine (GPU) | PaddleOCR | Text recognition with angle correction |
| OCR Engine (CPU) | Tesseract | Text recognition — OEM 1, PSM 3 |
| Table Extractor | pdfplumber + tabula-py | Structured table detection and CSV export |
| Image Processor | OpenCV | Figure detection via contour analysis |
| Output Generator | python-docx | Compiled DOCX document creation |
| Frontend | Streamlit | Web interface with real-time progress tracking |

---

## Configuration

**PaddleOCR (GPU)**
```python
ocr = PaddleOCR(use_angle_cls=True, lang="en")
```

**Tesseract (CPU)**
```python
pytesseract.image_to_string(img, config='--oem 1 --psm 3 -l eng')
# OEM 1: LSTM engine — higher accuracy
# PSM 3: Fully automatic page segmentation
```

**Figure Detection Threshold**
```python
# In detect_figures() — adjust w and h to control sensitivity
if w > 100 and h > 100:
```
Increase thresholds to reduce false positives on dense documents. Decrease to capture smaller figures.

**CSV Export**
```python
df.to_csv(csv_buffer, index=False, escapechar='\\', quoting=1)
# quoting=1 = csv.QUOTE_ALL — wraps every cell in quotes
# escapechar handles remaining special character edge cases
```

---

## Known Issues and Fixes Applied

| Issue | Cause | Fix Applied |
|---|---|---|
| `_csv.Error: need to escape` | Special characters in table cells | Added `escapechar='\\', quoting=1` to all `df.to_csv()` calls |
| `ValueError: XML NULL bytes` | Unsanitized OCR output written to DOCX | `sanitize_text()` applied to all table cell and header values before DOCX write |
| Inflated timing display | Accumulating timer inside loop | Fixed to single `time.time() - start` measurement at end of pipeline |

---

## Design Decisions

**Why parallel extraction?**
Text, table, and figure extraction are independent operations per page. ThreadPoolExecutor allows all three to run concurrently, reducing total processing time proportional to available CPU cores.

**Why pdfplumber alongside tabula-py?**
pdfplumber handles complex table boundary detection better on irregular layouts. tabula-py is retained as a complementary fallback for edge cases.

**Why DOCX as the combined output format?**
DOCX preserves embedded images alongside text and tables in a single portable file — more useful for downstream human review than a ZIP archive of separate files.

**Why two separate files instead of one with a toggle?**
PaddleOCR and Tesseract have different installation footprints and dependency chains. Keeping them in separate files means users can clone the repo and run only the version their hardware supports without installing unnecessary dependencies.

---

## Troubleshooting

**Missing text in tables:** Ensure the PDF is not a scanned image-only document. pdfplumber requires selectable text for table extraction.

**Poor figure detection:** Adjust contour size thresholds in `detect_figures()` for documents with small diagrams or dense layouts.

**Memory issues on large PDFs:** For files exceeding 50 pages, consider processing in chunks rather than loading all pages into memory at once.

**OCR accuracy on low-resolution scans:** Pre-process images with OpenCV binarization before passing to the OCR engine:
```python
gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
```

**PaddleOCR GPU not detected:** Verify CUDA installation with `nvidia-smi` and confirm paddlepaddle-gpu is installed instead of the CPU build.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

MIT License — see LICENSE file for details.

---

## Acknowledgments

PaddlePaddle for the OCR engine, Tesseract OCR for the open-source OCR implementation, Streamlit for the web framework, and all contributing libraries and their maintainers.
