import os
from unstructured.partition.pdf import partition_pdf

# Ensure Tesseract is in the PATH if using OCR (installation might handle this)
# For macOS with Homebrew: os.environ["PATH"] += ":/opt/homebrew/bin" 
# Or specify tesseract path directly in partition_pdf call if needed

file_path = "/Users/mau/Dev/lie_detector_agent/Paul_Ekman_Source_1.pdf"

# Partition the PDF using unstructured. 
# Set hi_res_model_name="config" to potentially improve accuracy with layout detection.
# Set infer_table_structure=True if you expect tables.
# Set ocr_languages if needed (e.g., 'eng')
elements = partition_pdf(
    filename=file_path, 
    strategy="ocr_only", # Uncomment this if you *know* it's image-based
    # hi_res_model_name="config", # Optional: might improve layout detection
    # infer_table_structure=True, # Optional: if you expect tables
    # ocr_languages="eng", # Optional: specify language(s)
)

# Extract and print text from the elements
extracted_text = "\n\n".join([str(el) for el in elements])

if extracted_text:
    print("Successfully extracted text using unstructured. First 1000 characters:")
    print(extracted_text[:1000])
else:
    print("No text could be extracted using unstructured.")

# Optional: Save the extracted text to a file
# with open("extracted_output.txt", "w") as f:
#     f.write(extracted_text)
