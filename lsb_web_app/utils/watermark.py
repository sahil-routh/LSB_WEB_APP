from PIL import Image
import numpy as np
import os

def embed_extract(cover_path, watermark_path, output_dir):
    cover = Image.open(cover_path).convert("RGB")
    watermark = Image.open(watermark_path).convert("RGB")

    if cover.size != watermark.size:
        watermark = watermark.resize(cover.size)

    cover_arr = np.array(cover)
    watermark_arr = np.array(watermark)

    c_flat = cover_arr.flatten()
    w_flat = watermark_arr.flatten()

    bits = (w_flat >> 4) & 0x0F
    embedded = (c_flat & 0xF0) | bits

    watermarked = embedded.reshape(cover_arr.shape)
    wm_path = os.path.join(output_dir, "watermarked.png")
    Image.fromarray(watermarked.astype(np.uint8)).save(wm_path)

    # Extract from watermarked image
    extracted_bits_wm = watermarked.flatten() & 0x0F
    extracted_pixels_wm = (extracted_bits_wm << 4).reshape(cover_arr.shape)
    ext_wm_path = os.path.join(output_dir, "extracted_watermarked.png")
    Image.fromarray(extracted_pixels_wm.astype(np.uint8)).save(ext_wm_path)

    # Extract from cover image (for comparison)
    extracted_bits_cover = cover_arr.flatten() & 0x0F
    extracted_pixels_cover = (extracted_bits_cover << 4).reshape(cover_arr.shape)
    ext_cover_path = os.path.join(output_dir, "extracted_cover.png")
    Image.fromarray(extracted_pixels_cover.astype(np.uint8)).save(ext_cover_path)

    return wm_path, ext_wm_path, ext_cover_path