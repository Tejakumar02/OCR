Smart PDF Extractor with OCR & AI
📋 Overview
Smart PDF Extractor is a comprehensive document processing tool that extracts text, tables, figures, and links from PDF documents using advanced OCR technology. The application supports both PaddleOCR (GPU-optimized) and Tesseract (CPU-based) engines for flexible deployment options.

# install requirements.txt
# pip install -r requirements.txt
✨ Features
🔤 Text Extraction
Dual OCR Engines: Choose between PaddleOCR (GPU) and Tesseract (CPU)

Multi-language Support: Primary support for English with extensible language options

Formula Conversion: Automatic conversion of mathematical formulas to LaTeX format

URL Detection: Intelligent extraction of web links from text content

📊 Table Extraction
Advanced table detection using pdfplumber and tabula-py

Multi-page table extraction with page-wise organization

Export tables to CSV format with pandas integration

🖼️ Figure/Image Detection
Automatic figure detection using OpenCV contour analysis

Cropped figure extraction with size filtering

PNG format export for extracted images

🔗 Link Extraction
Comprehensive URL pattern matching

Structured PDF link extraction via PyMuPDF

Combined text and structured link detection

📁 Multi-format Support
Input Formats: PDF, PNG, JPG, JPEG, TXT, MD, DOCX

Output Formats: Markdown, CSV, PNG, DOCX, ZIP archives

Batch Processing: Simultaneous extraction of all content types

🚀 Performance Optimizations
Parallel processing with ThreadPoolExecutor

Progress tracking with real-time status updates

Memory-efficient streaming for large files

🛠️ Installation
Prerequisites
Python 3.8 or higher

Tesseract OCR installed on system (for CPU version)

Method 1: PaddleOCR Version (GPU Recommended)
bash
# Clone the repository
git clone <your-repo-url>
cd smart-pdf-extractor

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PaddleOCR requirements
pip install streamlit paddleocr paddlepaddle Pillow PyMuPDF opencv-python numpy pdfplumber tabula-py python-docx pandas
Method 2: Tesseract Version (CPU)
bash
# Install Tesseract requirements
pip install streamlit pytesseract pillow pymupdf opencv-python numpy tabula-py pdfplumber pandas python-docx

# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
🚀 Usage
Running the Application
bash
# For PaddleOCR version (GPU)
streamlit run main.py

# For Tesseract version (CPU)
streamlit run app.py
User Interface
Upload Document: Click the upload area to select PDF, image, or text files

Content Extraction: Press "Extract All Content" to process the document

Preview Results: View extracted text, tables, figures, and links in the interface

Download Options: Choose specific content types or download everything as a DOCX file

Output Options
📄 Text: Markdown format with page separation

📊 Tables: CSV files packaged in ZIP archives

🖼️ Figures: PNG images in ZIP archives

🔗 Links: Text files with extracted URLs

📦 All Content: Combined DOCX document with all extracted content

🏗️ Architecture

Key Components
Document Parser: PyMuPDF for PDF processing

OCR Engine: PaddleOCR/Tesseract for text recognition

Table Extractor: pdfplumber and tabula-py for structured data

Image Processor: OpenCV for figure detection and cropping

Output Generator: python-docx for document creation

Processing Pipeline
text
PDF Upload → Page Conversion → Parallel Extraction → Content Organization → Export
     ↓             ↓                 ↓                     ↓                 ↓
  File I/O    Image Rendering  Text/Tables/Figures    Data Structuring   Format Export

⚙️ Configuration
OCR Settings
python

# PaddleOCR Configuration
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)

# Tesseract Configuration
pytesseract.image_to_string(img, config='--oem 1 --psm 3 -l eng')
Performance Tuning
Adjust ThreadPoolExecutor max_workers based on system resources

Modify OpenCV contour detection thresholds for figure sensitivity

Configure pdfplumber table extraction parameters for accuracy

📊 Performance Metrics
Metric	PaddleOCR (GPU)	Tesseract (CPU)
Text Extraction Speed	⚡ Very Fast	🐢 Moderate
Accuracy	🎯 High	🎯 High
Memory Usage	📈 Higher	📉 Lower
Setup Complexity	🛠️ Moderate	🛠️ Simple

🐛 Known Issues & Solutions
Common Problems
Missing Text in Tables: Enable pdfplumber table extraction as fallback

Poor Figure Detection: Adjust contour size thresholds in detect_figures()

Memory Issues with Large PDFs: Implement chunk-based processing

OCR Accuracy Issues: Pre-process images with OpenCV (binarization, denoising)

Troubleshooting
python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check OCR installation
import pytesseract
print(pytesseract.get_tesseract_version())
🔧 Advanced Features
Custom Processing
python
# Add custom preprocessing
def preprocess_image(img):
    # Add your custom image preprocessing here
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
Extending Language Support
python
# For PaddleOCR
ocr = PaddleOCR(lang="ch", use_angle_cls=True)  # Chinese

# For Tesseract
pytesseract.image_to_string(img, lang='eng+fra+spa')  # Multiple languages
📈 Future Enhancements
Cloud storage integration (Google Drive, Dropbox)

API endpoint for programmatic access

Batch processing for multiple documents

Custom template-based extraction

Machine learning for improved classification

Real-time collaboration features

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
PaddlePaddle for OCR engine

Tesseract OCR for open-source OCR

Streamlit for the web framework

All contributing libraries and their maintainers

📞 Support
For issues, questions, or feature requests:

Check existing Issues

Create a new issue with detailed description

Include sample files for reproduction when possible

Note: GPU acceleration requires CUDA-compatible hardware and proper driver installation for PaddleOCR version. The Tesseract version is recommended for systems without GPU capabilities.
