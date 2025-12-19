from PIL import Image
import numpy as np
import os

def embed_extract_1bit(cover_path, watermark_path, output_dir):
    """
    One-bit LSB watermarking - embeds only 1 bit per pixel
    More imperceptible but requires larger cover image
    """
    cover = Image.open(cover_path).convert("RGB")
    watermark = Image.open(watermark_path).convert("1")  # Convert to binary (black/white)
    
    # For 1-bit LSB, cover needs to be at least as large as watermark
    # Each watermark pixel needs 1 cover pixel (1 bit per pixel)
    cover_pixels = cover.size[0] * cover.size[1]
    watermark_pixels = watermark.size[0] * watermark.size[1]
    
    if cover_pixels < watermark_pixels:
        raise ValueError(f"Cover image too small! Need at least {watermark_pixels} pixels, got {cover_pixels}")
    
    cover_arr = np.array(cover)
    watermark_arr = np.array(watermark)
    
    # Flatten arrays
    cover_flat = cover_arr.flatten()
    watermark_flat = watermark_arr.flatten()
    
    # Convert watermark to bits (0 or 1)
    watermark_bits = (watermark_flat > 127).astype(np.uint8)
    
    # Embed watermark bits into LSB of cover
    # Clear LSB of cover pixels and set to watermark bit
    embedded = cover_flat.copy()
    embedded[:len(watermark_bits)] = (embedded[:len(watermark_bits)] & 0xFE) | watermark_bits
    
    # Reshape back to image
    watermarked = embedded.reshape(cover_arr.shape)
    wm_path = os.path.join(output_dir, "watermarked_1bit.png")
    Image.fromarray(watermarked.astype(np.uint8)).save(wm_path)
    
    # Extract from watermarked image
    watermarked_flat = watermarked.flatten()
    extracted_bits = (watermarked_flat[:len(watermark_bits)] & 0x01) * 255
    
    # Pad to original watermark size
    extracted_padded = np.zeros_like(watermark_flat)
    extracted_padded[:len(extracted_bits)] = extracted_bits
    
    extracted_pixels = extracted_padded.reshape(watermark_arr.shape)
    ext_wm_path = os.path.join(output_dir, "extracted_watermarked_1bit.png")
    Image.fromarray(extracted_pixels.astype(np.uint8)).save(ext_wm_path)
    
    # Extract from cover image (for comparison)
    cover_flat_original = cover_arr.flatten()
    extracted_bits_cover = (cover_flat_original[:len(watermark_bits)] & 0x01) * 255
    
    extracted_padded_cover = np.zeros_like(watermark_flat)
    extracted_padded_cover[:len(extracted_bits_cover)] = extracted_bits_cover
    
    extracted_pixels_cover = extracted_padded_cover.reshape(watermark_arr.shape)
    ext_cover_path = os.path.join(output_dir, "extracted_cover_1bit.png")
    Image.fromarray(extracted_pixels_cover.astype(np.uint8)).save(ext_cover_path)
    
    return wm_path, ext_wm_path, ext_cover_path


def embed_extract_1bit_rgb(cover_path, watermark_path, output_dir):
    """
    One-bit LSB watermarking for RGB channels
    Embeds 1 bit in each RGB channel (3 bits total per pixel)
    Automatically resizes watermark to match cover size
    """
    cover = Image.open(cover_path).convert("RGB")
    watermark = Image.open(watermark_path).convert("RGB")
    
    # IMPORTANT: Resize watermark to match cover size
    if cover.size != watermark.size:
        watermark = watermark.resize(cover.size, Image.Resampling.LANCZOS)
    
    cover_arr = np.array(cover)
    watermark_arr = np.array(watermark)
    
    # Now both arrays have the same shape
    h, w, c = cover_arr.shape
    
    # Embed 1 bit per channel
    embedded = cover_arr.copy()
    for channel in range(3):  # R, G, B
        cover_channel = cover_arr[:, :, channel]
        watermark_channel = watermark_arr[:, :, channel]
        
        # Get MSB of watermark (most significant bit for better quality)
        watermark_bits = (watermark_channel >> 7) & 0x01
        
        # Clear LSB and embed watermark bit
        embedded[:, :, channel] = (cover_channel & 0xFE) | watermark_bits
    
    wm_path = os.path.join(output_dir, "watermarked_1bit_rgb.png")
    Image.fromarray(embedded.astype(np.uint8)).save(wm_path)
    
    # Extract from watermarked image
    extracted = np.zeros((h, w, c), dtype=np.uint8)
    for channel in range(3):
        embedded_channel = embedded[:, :, channel]
        # Extract LSB and shift to MSB position
        extracted[:, :, channel] = (embedded_channel & 0x01) << 7
    
    ext_wm_path = os.path.join(output_dir, "extracted_watermarked_1bit_rgb.png")
    Image.fromarray(extracted.astype(np.uint8)).save(ext_wm_path)
    
    # Extract from cover (for comparison)
    extracted_cover = np.zeros((h, w, c), dtype=np.uint8)
    for channel in range(3):
        cover_channel = cover_arr[:, :, channel]
        extracted_cover[:, :, channel] = (cover_channel & 0x01) << 7
    
    ext_cover_path = os.path.join(output_dir, "extracted_cover_1bit_rgb.png")
    Image.fromarray(extracted_cover.astype(np.uint8)).save(ext_cover_path)
    
    return wm_path, ext_wm_path, ext_cover_path