# Text Extraction From Videos #

The pipeline is as follows:
1. **Text Detection** : Drawing bounding boxes around the detected text
2. **Character Segmentation** : Segmenting the characters in the detected text
3. **Character Recognition** : Recognising individual characters

Methodology adopted:
1. **EAST** (An Efficient and Accurate Scene Text Detector) model for Text Detection
   It is a scene text detector that directly produces word or line level predictions from full images with
   a single neural network.<br/>
   Link to the paper: https://arxiv.org/pdf/1704.03155.pdf

2. **Pytesseract** for Recognition
   The output bounding box of the detected text is localized and given as input to the pytesseract tool.
   Pytesseract is a powerful OCR tool used to recognize text.
   
Instructions to run:<br/>
Required libraries are OpenCV, pytesseract, imutils<br/>
The repository contains test videos to test the code

Download **frozen_east_text_detection.pb**, **test_video** for the EAST model and test video respectively<br/>
Command to run : python3 text_extraction_from_video.py

