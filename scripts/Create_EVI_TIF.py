import numpy as np
from osgeo import gdal
import os

# ======================
# USER INPUT PARAMETERS
# ======================
input_folder = "data/Neiva-2024-04-10"
output_folder = "output"

# Band file names (modify according to your Landsat files)
blue_band = os.path.join(input_folder, "SR_B2.TIF")     # Band 2 (Blue)
red_band = os.path.join(input_folder, "SR_B4.TIF")      # Band 4 (Red)
nir_band = os.path.join(input_folder, "SR_B5.TIF")      # Band 5 (NIR)
qa_band = os.path.join(input_folder, "QA_PIXEL.TIF") # QA_PIXEL band

# ======================
# PROCESSING FUNCTIONS
# ======================

def read_raster(path):
    """Read a raster file and return array and metadata"""
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    no_data = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    prj = ds.GetProjection()
    return arr, gt, prj, no_data

def write_raster(path, array, gt, prj, no_data=None, dtype=gdal.GDT_Float32):
    """Write array to new raster file"""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    out_ds = driver.Create(path, cols, rows, 1, dtype)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(prj)
    out_band = out_ds.GetRasterBand(1)
    if no_data is not None:
        out_band.SetNoDataValue(no_data)
    out_band.WriteArray(array)
    out_band.FlushCache()
    out_ds = None

def create_cloud_mask(qa_array):
    """Create cloud mask from QA_PIXEL array (1=clear, 0=cloud/shadow)"""
    # Cloud (Bit 3) and Cloud Shadow (Bit 4) = 8 + 16 = 24
    return (qa_array & 24) == 0

def calculate_evi(blue_arr, red_arr, nir_arr, mask_arr, scale=10000.0):
    """Calculate EVI using scaled reflectance and cloud mask"""
    # Scale reflectance values
    blue = blue_arr.astype(np.float32) / scale
    red = red_arr.astype(np.float32) / scale
    nir = nir_arr.astype(np.float32) / scale
    
    # Calculate EVI only where mask is True (1)
    evi = np.zeros_like(blue, dtype=np.float32)
    valid = mask_arr == 1
    
    # EVI formula
    evi[valid] = 2.5 * (nir[valid] - red[valid]) / (
        nir[valid] + 6 * red[valid] - 7.5 * blue[valid] + 1
    )
    
    # Clip to valid range (-1 to 1)
    evi = np.clip(evi, -1, 1)
    return evi

# ======================
# MAIN PROCESSING
# ======================

def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print("Loading input bands...")
    # Read all input rasters
    blue_arr, gt, prj, _ = read_raster(blue_band)
    red_arr, _, _, _ = read_raster(red_band)
    nir_arr, _, _, _ = read_raster(nir_band)
    qa_arr, _, _, _ = read_raster(qa_band)
    
    print("Creating cloud mask...")
    mask_arr = create_cloud_mask(qa_arr)
    mask_path = os.path.join(output_folder, "cloud_mask_Neiva.tif")
    write_raster(mask_path, mask_arr.astype(np.uint8), gt, prj, no_data=0)
    
    print("Calculating EVI...")
    evi_arr = calculate_evi(blue_arr, red_arr, nir_arr, mask_arr)
    evi_path = os.path.join(output_folder, "EVI_Neiva.tif")
    write_raster(evi_path, evi_arr, gt, prj, no_data=-9999)
    
    print(f"Processing complete! Output saved to: {output_folder}")

if __name__ == "__main__":
    main()