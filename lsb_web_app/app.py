from flask import Flask, render_template, request
import os
from utils.watermark import embed_extract  # 4-bit LSB
from utils.watermark_1bit import embed_extract_1bit_rgb  # 1-bit LSB RGB
from utils.metrics import mse, psnr
from PIL import Image
import numpy as np
import shutil

app = Flask(__name__)

UPLOAD = "static/uploads"
OUTPUT = "static/outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    cover_file = request.files["cover"]
    watermark_file = request.files["watermark"]
    lsb_method = request.form.get("lsb_method", "4bit")  # Default to 4-bit
    
    cover_path = os.path.join(UPLOAD, cover_file.filename)
    watermark_path = os.path.join(UPLOAD, watermark_file.filename)

    cover_file.save(cover_path)
    watermark_file.save(watermark_path)

    # Choose embedding method based on selection
    if lsb_method == "1bit":
        watermarked_path, extracted_wm_path, extracted_cover_path = embed_extract_1bit_rgb(
            cover_path, watermark_path, OUTPUT
        )
        method_name = "1-bit LSB (per RGB channel)"
    else:  # 4bit
        watermarked_path, extracted_wm_path, extracted_cover_path = embed_extract(
            cover_path, watermark_path, OUTPUT
        )
        method_name = "4-bit LSB"

    # Load images for metrics
    cover = np.array(Image.open(cover_path).convert("L"))
    watermarked = np.array(Image.open(watermarked_path).convert("L"))
    extracted_cover = np.array(Image.open(extracted_cover_path).convert("L"))
    extracted_wm = np.array(Image.open(extracted_wm_path).convert("L"))

    result = {
        "watermarked": watermarked_path,
        "extracted_cover": extracted_cover_path,
        "extracted_watermarked": extracted_wm_path,
        "mse1": round(mse(extracted_wm, watermarked), 4),      # Extracted vs Watermarked image 
        "psnr1": round(psnr(extracted_wm, watermarked), 2),    # Extracted vs Watermarked image
        "mse2": round(mse(extracted_cover, cover), 4),         # Extracted vs Cover image 
        "psnr2": round(psnr(extracted_cover, cover), 2),       # Extracted vs Cover image 
        "method": method_name
    }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)