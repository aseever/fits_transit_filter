import os
import json
import numpy as np
from datetime import datetime, timedelta
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import glob

def find_image_sets(data_dir, time_min_hours=1, time_max_hours=48, 
                    position_threshold_arcmin=10):
    """
    Find sets of FITS images taken at the same sky location between time_min_hours
    and time_max_hours apart.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing FITS files
    time_min_hours : float
        Minimum time difference in hours
    time_max_hours : float
        Maximum time difference in hours
    position_threshold_arcmin : float
        Maximum separation in arcminutes to consider as the "same" sky location
        
    Returns:
    --------
    list
        List of dictionaries containing image sets
    """
    # Get all FITS files in the directory
    fits_files = []
    for extension in ['*.fits', '*.fit', '*.fts']:
        fits_files.extend(glob.glob(os.path.join(data_dir, '**', extension), recursive=True))
    
    if not fits_files:
        print("No FITS files found in directory.")
        return []
    
    # Extract metadata from each file
    image_data = []
    for file_path in fits_files:
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                
                # Extract coordinates (RA and DEC)
                if 'RA' in header and 'DEC' in header:
                    ra = header['RA']
                    dec = header['DEC']
                elif 'CRVAL1' in header and 'CRVAL2' in header:
                    ra = header['CRVAL1']
                    dec = header['CRVAL2']
                else:
                    print(f"Warning: Could not find coordinates in {file_path}")
                    continue
                
                # Extract observation time
                if 'DATE-OBS' in header:
                    date_obs = header['DATE-OBS']
                    # Handle different date formats
                    try:
                        if 'T' in date_obs:
                            obs_time = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S.%f')
                        else:
                            obs_time = datetime.strptime(date_obs, '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try:
                            if 'T' in date_obs:
                                obs_time = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S')
                            else:
                                obs_time = datetime.strptime(date_obs, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            print(f"Warning: Could not parse observation time in {file_path}")
                            continue
                else:
                    print(f"Warning: No observation time found in {file_path}")
                    continue
                
                # Extract image quality metrics if available
                dead_pixels = header.get('DEADPIX', None)
                seeing = header.get('SEEING', None)
                
                image_data.append({
                    'file_path': file_path,
                    'ra': float(ra),
                    'dec': float(dec),
                    'obs_time': obs_time,
                    'dead_pixels': dead_pixels,
                    'seeing': seeing
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Group images by pointing
    pointing_groups = {}
    
    for i, img1 in enumerate(image_data):
        coord1 = SkyCoord(ra=img1['ra']*u.degree, dec=img1['dec']*u.degree)
        
        for j, img2 in enumerate(image_data[i+1:], i+1):
            coord2 = SkyCoord(ra=img2['ra']*u.degree, dec=img2['dec']*u.degree)
            
            # Calculate angular separation
            sep = coord1.separation(coord2).arcmin
            
            if sep <= position_threshold_arcmin:
                # Images are at the same location
                pointing_key = f"{img1['ra']:.3f}_{img1['dec']:.3f}"
                
                if pointing_key not in pointing_groups:
                    pointing_groups[pointing_key] = []
                
                # Add img1 if it's not already in the group
                if not any(img['file_path'] == img1['file_path'] for img in pointing_groups[pointing_key]):
                    pointing_groups[pointing_key].append(img1)
                
                # Add img2 if it's not already in the group
                if not any(img['file_path'] == img2['file_path'] for img in pointing_groups[pointing_key]):
                    pointing_groups[pointing_key].append(img2)
    
    # Find sets of images within the time window
    image_sets = []
    
    for pointing, images in pointing_groups.items():
        # Sort images by observation time
        images.sort(key=lambda x: x['obs_time'])
        
        # Check for pairs or groups within time window
        valid_sets = []
        
        for i, img1 in enumerate(images):
            set_group = [img1]
            
            for img2 in images[i+1:]:
                time_diff = (img2['obs_time'] - img1['obs_time']).total_seconds() / 3600.0
                
                if time_min_hours <= time_diff <= time_max_hours:
                    set_group.append(img2)
            
            if len(set_group) > 1:
                valid_sets.append(set_group)
        
        # Add valid sets to results
        for valid_set in valid_sets:
            image_set = {
                'pointing': {
                    'ra': valid_set[0]['ra'],
                    'dec': valid_set[0]['dec']
                },
                'images': [
                    {
                        'file_path': img['file_path'],
                        'obs_time': img['obs_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        'dead_pixels': img['dead_pixels'],
                        'seeing': img['seeing']
                    } for img in valid_set
                ],
                'time_span_hours': (valid_set[-1]['obs_time'] - valid_set[0]['obs_time']).total_seconds() / 3600.0
            }
            image_sets.append(image_set)
    
    return image_sets

def main():
    data_directory = '/data'  # Change this if needed
    output_file = 'kbo_image_sets.json'
    
    # Find image sets
    print(f"Scanning directory: {data_directory}")
    image_sets = find_image_sets(data_directory)
    
    print(f"Found {len(image_sets)} sets of images suitable for KBO detection")
    
    # Convert datetime objects to strings for JSON serialization
    for image_set in image_sets:
        for image in image_set['images']:
            if isinstance(image['obs_time'], datetime):
                image['obs_time'] = image['obs_time'].strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(image_sets, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()