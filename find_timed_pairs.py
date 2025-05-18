import os
import json
import re
from datetime import datetime, timedelta
from astropy.io import fits
import glob

def find_image_sets(data_dir, time_min_hours=0, time_max_hours=72):
    """
    Find sets of HST FITS images taken between time_min_hours
    and time_max_hours apart.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing FITS files
    time_min_hours : float
        Minimum time difference in hours (set to 0 to include same-day observations)
    time_max_hours : float
        Maximum time difference in hours (increased to 72 for more flexibility)
        
    Returns:
    --------
    list
        List of dictionaries containing image sets
    """
    # Get all FITS files in the directory
    fits_files = []
    for extension in ['*.fits', '*.fit', '*.fts']:
        pattern = os.path.join(data_dir, '**', extension)
        print(f"Searching for pattern: {pattern}")
        files = glob.glob(pattern, recursive=True)
        fits_files.extend(files)
        print(f"Found {len(files)} files with extension {extension}")
    
    print(f"Found a total of {len(fits_files)} FITS files")
    
    if not fits_files:
        print("No FITS files found in directory.")
        return []
    
    # Extract metadata from each file
    image_data = []
    
    # Regular expression for HST file naming pattern (e.g., icii01drq)
    hst_pattern = re.compile(r'(ic[a-z]{2}\d{2}[a-z0-9]{3})')
    
    for file_path in fits_files:
        try:
            # Extract HST visit ID from filename
            visit_id = None
            hst_match = hst_pattern.search(os.path.basename(file_path))
            if hst_match:
                visit_id = hst_match.group(1)
            
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                
                # Try different date keywords used in HST data
                obs_time = None
                date_keywords = ['DATE-OBS', 'EXPSTART', 'DATE', 'EXPEND', 'DATE-BEG']
                
                for keyword in date_keywords:
                    if keyword in header:
                        try:
                            date_value = header[keyword]
                            
                            # Convert MJD format to datetime if necessary
                            if isinstance(date_value, (float, int)) and 40000 < date_value < 70000:
                                # Likely Modified Julian Date
                                mjd_zero = datetime(1858, 11, 17)
                                obs_time = mjd_zero + timedelta(days=date_value)
                                break
                            
                            # Try various date formats
                            date_formats = [
                                '%Y-%m-%dT%H:%M:%S.%f', 
                                '%Y-%m-%d %H:%M:%S.%f',
                                '%Y-%m-%dT%H:%M:%S', 
                                '%Y-%m-%d %H:%M:%S',
                                '%Y-%m-%d'
                            ]
                            
                            for date_format in date_formats:
                                try:
                                    obs_time = datetime.strptime(date_value, date_format)
                                    break
                                except (ValueError, TypeError):
                                    continue
                            
                            if obs_time:
                                break
                                
                        except Exception as e:
                            pass  # Silently try the next keyword
                
                if obs_time is None:
                    # As a last resort, try to infer date from filename or directory structure
                    file_mtime = os.path.getmtime(file_path)
                    obs_time = datetime.fromtimestamp(file_mtime)
                
                # Extract target information
                target = header.get('TARGNAME', None)
                if not target:
                    target = header.get('OBJECT', None)
                
                # Extract program information
                program_id = header.get('PROPOSID', None)
                
                # Extract filter information
                filter_name = header.get('FILTER', None)
                if not filter_name:
                    filter_name = header.get('FILTER1', None)
                
                image_data.append({
                    'file_path': file_path,
                    'visit_id': visit_id,  # HST visit ID
                    'obs_time': obs_time,
                    'target': target,
                    'program_id': program_id,
                    'filter': filter_name
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Successfully extracted metadata from {len(image_data)} of {len(fits_files)} files")
    
    # Filter out entries with no observation time
    valid_images = [img for img in image_data if img['obs_time'] is not None]
    print(f"Files with valid observation times: {len(valid_images)}")
    
    if len(valid_images) == 0:
        print("No files with valid observation times found.")
        return []
    
    # Print timeline of observations
    valid_images.sort(key=lambda x: x['obs_time'])
    print("Observation Timeline:")
    prev_date = None
    for i, img in enumerate(valid_images, 1):
        current_date = img['obs_time'].date()
        days_diff = f"({(current_date - prev_date).days} days since previous)" if prev_date else ""
        print(f"{i}: {current_date} - Visit {img['visit_id']} {days_diff}")
        prev_date = current_date
        if i == 20 and len(valid_images) > 20:
            print(f"... and {len(valid_images) - 20} more observations")
            break
    
    # Group images in two ways:
    # 1. By exact target
    target_groups = {}
    for img in valid_images:
        if img['target']:
            if img['target'] not in target_groups:
                target_groups[img['target']] = []
            target_groups[img['target']].append(img)
    
    # 2. By target region (extract row and column from target name)
    region_pattern = re.compile(r'KBO\d+\-R(\d+)C(\d+)')
    region_groups = {}
    
    for img in valid_images:
        if img['target']:
            match = region_pattern.match(img['target'])
            if match:
                row, col = match.groups()
                # Group by row only, allowing adjacent columns
                region_key = f"R{row}"
                if region_key not in region_groups:
                    region_groups[region_key] = []
                region_groups[region_key].append(img)
    
    print(f"Found {len(target_groups)} unique targets and {len(region_groups)} unique regions")
    
    # Look for image pairs within the time window
    print("\nLooking for image pairs by exact target...")
    
    time_window_pairs = []
    
    # Find pairs within each target group
    for target, target_images in target_groups.items():
        if len(target_images) > 1:
            print(f"Processing target: {target} with {len(target_images)} images")
            
            # Sort by observation time
            target_images.sort(key=lambda x: x['obs_time'])
            
            for i, img1 in enumerate(target_images):
                for j, img2 in enumerate(target_images[i+1:], i+1):
                    # Calculate time difference in hours
                    time_diff = (img2['obs_time'] - img1['obs_time']).total_seconds() / 3600.0
                    
                    # Check if within time window
                    if time_min_hours <= time_diff <= time_max_hours:
                        time_window_pairs.append((img1, img2, time_diff, "exact_target"))
                        print(f"  Found pair: {img1['visit_id']} and {img2['visit_id']} ({time_diff:.2f} hours apart)")
    
    # Now look for pairs by region
    print("\nLooking for image pairs by region (same row, adjacent columns)...")
    
    for region, region_images in region_groups.items():
        if len(region_images) > 1:
            print(f"Processing region: {region} with {len(region_images)} images")
            
            # Sort by observation time
            region_images.sort(key=lambda x: x['obs_time'])
            
            for i, img1 in enumerate(region_images):
                for j, img2 in enumerate(region_images[i+1:], i+1):
                    # Skip if they're the same target
                    if img1['target'] == img2['target']:
                        continue
                    
                    # Check if they're adjacent columns in the same row
                    match1 = region_pattern.match(img1['target'])
                    match2 = region_pattern.match(img2['target'])
                    
                    if match1 and match2:
                        _, col1 = match1.groups()
                        _, col2 = match2.groups()
                        
                        if abs(int(col1) - int(col2)) <= 1:  # Adjacent columns
                            # Calculate time difference in hours
                            time_diff = (img2['obs_time'] - img1['obs_time']).total_seconds() / 3600.0
                            
                            # Check if within time window
                            if time_min_hours <= time_diff <= time_max_hours:
                                time_window_pairs.append((img1, img2, time_diff, "adjacent_region"))
                                print(f"  Found region pair: {img1['target']} and {img2['target']} ({time_diff:.2f} hours apart)")
    
    print(f"\nTotal image pairs within time window ({time_min_hours}-{time_max_hours} hours): {len(time_window_pairs)}")
    
    # Create image sets from the pairs
    image_sets = []
    
    for img1, img2, time_diff, pair_type in time_window_pairs:
        if not img1['visit_id'] or not img2['visit_id']:
            print(f"Warning: Skipping pair with missing visit ID: {img1['visit_id']} and {img2['visit_id']}")
            continue
            
        image_set = {
            'visits': [img1['visit_id'], img2['visit_id']],
            'images': [
                {
                    'file_path': img1['file_path'],
                    'visit_id': img1['visit_id'],
                    'obs_time': img1['obs_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'target': img1['target'],
                    'filter': img1['filter']
                },
                {
                    'file_path': img2['file_path'],
                    'visit_id': img2['visit_id'],
                    'obs_time': img2['obs_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'target': img2['target'],
                    'filter': img2['filter']
                }
            ],
            'time_diff_hours': time_diff,
            'target': f"{img1['target']} & {img2['target']}" if img1['target'] != img2['target'] else img1['target'],
            'pair_type': pair_type
        }
        image_sets.append(image_set)
    
    return image_sets

def main():
    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_file = 'kbo_image_sets.json'
    
    # Find image sets with more flexible time criteria
    print(f"Scanning directory: {data_directory}")
    image_sets = find_image_sets(data_directory, time_min_hours=0, time_max_hours=72)
    
    print(f"Found {len(image_sets)} sets of images suitable for KBO detection")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(image_sets, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()