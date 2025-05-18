import os
import json
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder
import time
from scipy.spatial import cKDTree
from itertools import combinations

def align_and_visualize_star_matches(json_path, output_dir, downsample_factor=4):
    """
    Align image pairs using centroid-based pattern matching.
    Very memory efficient, avoids correlation.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        image_sets = json.load(f)
    
    alignment_results = []
    
    # Process each image pair
    for i, image_set in enumerate(image_sets):
        start_time = time.time()
        print(f"\nProcessing image set {i+1}/{len(image_sets)}")
        time_diff = image_set['time_diff_hours']
        print(f"Time difference: {time_diff:.2f} hours")
        
        # Create a subdirectory for this image pair
        pair_dir = os.path.join(output_dir, f"pair_{i+1}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(pair_dir, "metadata.txt"), 'w') as f:
            f.write(f"Image Pair {i+1}\n")
            f.write(f"Skycells: {', '.join(image_set['skycells'])}\n")
            f.write(f"Time difference: {time_diff} hours\n")
            f.write(f"First image: {os.path.basename(image_set['images'][0]['file_path'])}, observed at {image_set['images'][0]['obs_time']}\n")
            f.write(f"Second image: {os.path.basename(image_set['images'][1]['file_path'])}, observed at {image_set['images'][1]['obs_time']}\n")
        
        try:
            # Load the first image
            img1_path = image_set['images'][0]['file_path'].replace('\\', os.sep)
            img1_basename = os.path.basename(img1_path)
            img1_time = image_set['images'][0]['obs_time']
            print(f"Reading first image: {img1_basename}")
            
            # Check if file exists or try alternative locations
            if not os.path.exists(img1_path):
                alt_path = os.path.join("data", img1_basename)
                if os.path.exists(alt_path):
                    img1_path = alt_path
                else:
                    print(f"Could not find image file: {img1_path}")
                    continue
            
            with fits.open(img1_path) as hdul1:
                # Use HDU 1 as we identified in debugging
                data1 = hdul1[1].data.astype(np.float32)
                header1 = hdul1[1].header
                
                # Downsample for faster processing
                if downsample_factor > 1:
                    data1 = data1[::downsample_factor, ::downsample_factor]
                    print(f"  Downsampled to shape: {data1.shape}")
            
            # Load the second image
            img2_path = image_set['images'][1]['file_path'].replace('\\', os.sep)
            img2_basename = os.path.basename(img2_path)
            img2_time = image_set['images'][1]['obs_time']
            print(f"Reading second image: {img2_basename}")
            
            # Check if file exists or try alternative locations
            if not os.path.exists(img2_path):
                alt_path = os.path.join("data", img2_basename)
                if os.path.exists(alt_path):
                    img2_path = alt_path
                else:
                    print(f"Could not find image file: {img2_path}")
                    continue
            
            with fits.open(img2_path) as hdul2:
                # Use HDU 1 as we identified in debugging
                data2 = hdul2[1].data.astype(np.float32)
                header2 = hdul2[1].header
                
                # Downsample for faster processing
                if downsample_factor > 1:
                    data2 = data2[::downsample_factor, ::downsample_factor]
            
            # Handle NaN values
            data1 = np.nan_to_num(data1)
            data2 = np.nan_to_num(data2)
            
            # Ensure images are the same size (crop if needed)
            if data1.shape != data2.shape:
                print(f"  Images have different shapes: {data1.shape} vs {data2.shape}")
                min_height = min(data1.shape[0], data2.shape[0])
                min_width = min(data1.shape[1], data2.shape[1])
                
                # Center crop
                def center_crop(img, target_height, target_width):
                    start_y = max(0, (img.shape[0] - target_height) // 2)
                    start_x = max(0, (img.shape[1] - target_width) // 2)
                    return img[start_y:start_y+target_height, start_x:start_x+target_width]
                
                data1 = center_crop(data1, min_height, min_width)
                data2 = center_crop(data2, min_height, min_width)
                print(f"  Cropped to shape: {data1.shape}")
            
            # Background estimation and subtraction
            print("  Background estimation and subtraction...")
            mean1, median1, std1 = sigma_clipped_stats(data1, sigma=3.0)
            data1_bg_sub = data1 - median1
            
            mean2, median2, std2 = sigma_clipped_stats(data2, sigma=3.0)
            data2_bg_sub = data2 - median2
            
            # Detect sources (stars) in both images
            print("  Detecting stars in both images...")
            
            # Find stars in first image - use a higher threshold for fewer but more reliable stars
            daofind1 = DAOStarFinder(fwhm=4.0, threshold=7.*std1)
            sources1 = daofind1(data1_bg_sub)
            if sources1 is None or len(sources1) == 0:
                print("  No stars detected in first image!")
                alignment_results.append({
                    'pair_id': i+1,
                    'alignment_success': False,
                    'reason': 'No stars detected in first image',
                    'match_count': 0,
                    'match_ratio': 0,
                    'rms_error': float('inf')
                })
                continue
                
            # Find stars in second image
            daofind2 = DAOStarFinder(fwhm=4.0, threshold=7.*std2)
            sources2 = daofind2(data2_bg_sub)
            if sources2 is None or len(sources2) == 0:
                print("  No stars detected in second image!")
                alignment_results.append({
                    'pair_id': i+1,
                    'alignment_success': False,
                    'reason': 'No stars detected in second image',
                    'match_count': 0,
                    'match_ratio': 0,
                    'rms_error': float('inf')
                })
                continue
            
            # Sort by flux (brightest first)
            sources1.sort('peak', reverse=True)
            sources2.sort('peak', reverse=True)
            
            print(f"  Found {len(sources1)} stars in first image and {len(sources2)} stars in second image")
            
            # Limit to the brightest stars for matching (memory efficient)
            max_stars = 30  # Use only the brightest stars for pattern matching
            sources1_bright = sources1[:min(max_stars, len(sources1))]
            sources2_bright = sources2[:min(max_stars, len(sources2))]
            
            # Extract coordinates of brightest stars
            coords1 = np.array([sources1_bright['xcentroid'], sources1_bright['ycentroid']]).T
            coords2 = np.array([sources2_bright['xcentroid'], sources2_bright['ycentroid']]).T
            
            # Extract coordinates of all stars for visualization later
            coords1_all = np.array([sources1['xcentroid'], sources1['ycentroid']]).T
            coords2_all = np.array([sources2['xcentroid'], sources2['ycentroid']]).T
            
            # We'll use a centroid-based pattern matching approach
            # This avoids correlation operations which are memory-intensive
            print("  Attempting alignment using centroid pattern matching...")
            
            # Try a simple approach first: use the center of mass of the brightest stars
            # This can work well if the fields are similar
            centroid1 = np.mean(coords1, axis=0)
            centroid2 = np.mean(coords2, axis=0)
            
            initial_offset = centroid1 - centroid2
            print(f"  Initial centroid offset: ({initial_offset[0]:.1f}, {initial_offset[1]:.1f}) pixels")
            
            # Let's test this offset along with a few other candidates
            offsets_to_try = [
                initial_offset,  # Centroid-based offset
                (0, 0),          # No offset (images already aligned)
                (initial_offset[0], 0),  # Just X offset
                (0, initial_offset[1]),  # Just Y offset
            ]
            
            # Add some small variations around the initial offset
            for dx in [-10, 0, 10]:
                for dy in [-10, 0, 10]:
                    if dx != 0 or dy != 0:  # Skip zeros we already have
                        offsets_to_try.append(initial_offset + np.array([dx, dy]))
            
            # Try each offset and find the one that gives the most matches
            best_offset = (0, 0)
            best_match_count = 0
            best_matches = []
            
            print(f"  Testing {len(offsets_to_try)} potential offsets...")
            
            # Use a slightly larger matching distance for initial tests
            matching_distance = 5.0  # pixels
            
            for offset in offsets_to_try:
                # Apply offset to coords2
                coords2_shifted = coords2 + offset
                
                # Build a KD-tree for efficient neighbor search
                tree1 = cKDTree(coords1)
                
                # Find the nearest star in the first image for each star in the shifted second image
                distances, indices = tree1.query(coords2_shifted, k=1, distance_upper_bound=matching_distance)
                
                # Count good matches (within matching distance)
                good_matches = distances < matching_distance
                match_count = np.sum(good_matches)
                
                # Extract the match pairs
                matches = [(indices[i], i) for i in range(len(distances)) if good_matches[i]]
                
                # If this is the best match so far, save it
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_offset = offset
                    best_matches = matches
                    
            x_shift, y_shift = best_offset
            
            print(f"  Best alignment offset: ({x_shift:.1f}, {y_shift:.1f}) pixels")
            print(f"  Matched stars with this offset: {best_match_count}")
            
            # Apply the best offset to all stars from the second image
            coords2_all_shifted = coords2_all + best_offset
            
            # Calculate quality metrics for the alignment
            match_count = len(best_matches)
            match_ratio = match_count / min(len(coords1), len(coords2))
            
            # Calculate alignment quality
            if match_count >= 3:
                # Calculate the RMS distance between matched stars
                matched_dists = []
                for idx1, idx2 in best_matches:
                    dist = np.sqrt(np.sum((coords1[idx1] - (coords2[idx2] + best_offset))**2))
                    matched_dists.append(dist)
                
                rms_error = np.sqrt(np.mean(np.array(matched_dists)**2))
                print(f"  RMS alignment error: {rms_error:.2f} pixels")
                
                # Determine alignment success based on match ratio and RMS error
                alignment_success = match_ratio > 0.3 and rms_error < 3.0
                
                if alignment_success:
                    print("  Alignment quality: GOOD")
                    reason = "Good star match"
                else:
                    if match_ratio <= 0.3:
                        print("  Alignment quality: POOR - too few matches")
                        reason = "Too few matches"
                    else:
                        print("  Alignment quality: POOR - high RMS error")
                        reason = "High RMS error"
            else:
                alignment_success = False
                rms_error = float('inf')
                print("  Alignment quality: FAILED - insufficient matches")
                reason = "Insufficient matches"
            
            alignment_results.append({
                'pair_id': i+1,
                'alignment_success': alignment_success,
                'reason': reason,
                'match_count': match_count,
                'match_ratio': match_ratio,
                'rms_error': rms_error if not np.isnan(rms_error) else float('inf')
            })
            
            # Generate visualization
            print("  Generating visualization...")
            
            # Normalize data for display
            def normalize_for_display(data, median, std):
                normalized = (data - median) / (5 * std)
                return np.clip(normalized, 0, 1)
            
            # Create normalized versions for display
            norm1 = normalize_for_display(data1, median1, std1)
            norm2 = normalize_for_display(data2, median2, std2)
            
            # Create RGB composite (red=img1, green=img2, blue=zeros)
            composite = np.zeros((data1.shape[0], data1.shape[1], 3))
            composite[:,:,0] = norm1  # Red channel from first image
            composite[:,:,1] = norm2  # Green channel from second image
            
            # Create figure
            plt.figure(figsize=(16, 12))
            
            # Show the composite image
            plt.subplot(2, 2, 1)
            plt.imshow(composite, origin='lower')
            plt.title(f"Composite (Red=Image1, Green=Image2)\nTime diff: {time_diff:.2f} hours", fontsize=12)
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            
            # Create the difference image for reference
            # First normalize the second image to match the first
            normalized_data2 = data2 * (std1 / std2)
            
            # Shift the second image by the calculated offset
            from scipy.ndimage import shift
            shifted_data2 = shift(normalized_data2, (y_shift, x_shift), mode='constant', cval=0)
            
            # Calculate the difference
            diff_image = data1 - shifted_data2
            
            # Show the first image with stars marked
            plt.subplot(2, 2, 2)
            plt.imshow(data1, origin='lower', cmap='gray', 
                      vmin=median1, vmax=median1+10*std1)
            plt.title(f"First Image: {img1_basename}\nDetected stars: {len(sources1)}", fontsize=12)
            
            # Mark all detected stars (limit for clarity)
            max_stars_to_plot = min(200, len(coords1_all))
            plt.scatter(coords1_all[:max_stars_to_plot,0], coords1_all[:max_stars_to_plot,1], 
                       s=30, facecolor='none', edgecolor='blue', alpha=0.5, label='All stars')
            
            # Mark matched stars
            if best_matches:
                match_coords1 = np.array([coords1[idx1] for idx1, _ in best_matches])
                plt.scatter(match_coords1[:,0], match_coords1[:,1], 
                          s=60, facecolor='none', edgecolor='yellow', label='Matched stars')
            
            plt.legend(loc='upper right')
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            
            # Show the second image with stars marked
            plt.subplot(2, 2, 3)
            plt.imshow(data2, origin='lower', cmap='gray', 
                      vmin=median2, vmax=median2+10*std2)
            plt.title(f"Second Image: {img2_basename}\nDetected stars: {len(sources2)}", fontsize=12)
            
            # Mark all detected stars (limit for clarity)
            max_stars_to_plot = min(200, len(coords2_all))
            plt.scatter(coords2_all[:max_stars_to_plot,0], coords2_all[:max_stars_to_plot,1], 
                       s=30, facecolor='none', edgecolor='blue', alpha=0.5, label='All stars')
            
            # Mark matched stars
            if best_matches:
                match_coords2 = np.array([coords2[idx2] for _, idx2 in best_matches])
                plt.scatter(match_coords2[:,0], match_coords2[:,1], 
                          s=60, facecolor='none', edgecolor='yellow', label='Matched stars')
            
            plt.legend(loc='upper right')
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            
            # Show the difference image
            plt.subplot(2, 2, 4)
            
            # Use a diverging colormap for difference
            vmax = 5 * np.std(diff_image[np.isfinite(diff_image)])
            plt.imshow(diff_image, origin='lower', cmap='coolwarm', 
                      vmin=-vmax, vmax=vmax)
            plt.title(f"Difference Image\nOffset: ({x_shift:.1f}, {y_shift:.1f}) pixels", fontsize=12)
            plt.xlabel("Pixel X")
            plt.ylabel("Pixel Y")
            
            # Add colorbar
            plt.colorbar(label='Difference (ADU)')
            
            # Add summary of alignment quality
            alignment_text = (
                f"Alignment Quality: {'GOOD' if alignment_success else 'POOR/FAILED'}\n"
                f"Matching stars: {match_count} / {min(len(coords1), len(coords2))} ({match_ratio:.1%})\n"
                f"RMS error: {rms_error:.2f} pixels\n"
                f"Issue: {reason if not alignment_success else 'None'}"
            )
            
            plt.figtext(0.5, 0.01, alignment_text, ha='center', fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.98])
            plt.suptitle(f"Image Pair {i+1}: Alignment Results", fontsize=14)
            
            plt.savefig(os.path.join(pair_dir, "alignment_results.png"), dpi=300)
            plt.close()
            
            # Create normalized difference image for KBO detection
            diff_norm = diff_image / (std1 + std2)
            
            # Save just the difference image with better scaling
            plt.figure(figsize=(12, 10))
            vmax = 3 * np.std(diff_norm[np.isfinite(diff_norm)])
            plt.imshow(diff_norm, origin='lower', cmap='coolwarm', vmin=-vmax, vmax=vmax)
            plt.title(f"Normalized Difference Image (Time diff: {time_diff:.2f} hours)", fontsize=14)
            plt.colorbar(label='Normalized Difference')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlabel("Pixel X", fontsize=12)
            plt.ylabel("Pixel Y", fontsize=12)
            
            # Add text about potential KBO interpretation
            plt.figtext(0.5, 0.01,
                       "Difference Image Interpretation:\n"
                       "• RED areas show objects brighter in first image, BLUE areas show objects brighter in second image\n"
                       "• Moving objects typically appear as RED-BLUE dipoles (nearby bright-dark pairs)\n"
                       "• For time difference of {:.2f} hours, KBOs would move by approximately 1-5 pixels".format(time_diff),
                       ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.07, 1, 0.98])
            plt.savefig(os.path.join(pair_dir, "difference_image.png"), dpi=300)
            plt.close()
            
            elapsed_time = time.time() - start_time
            print(f"  Completed processing image set {i+1} in {elapsed_time:.1f} seconds\n")
            
        except Exception as e:
            print(f"Error processing image set {i+1}: {e}")
            import traceback
            traceback.print_exc()
            alignment_results.append({
                'pair_id': i+1,
                'alignment_success': False,
                'reason': f'Error: {str(e)}',
                'match_count': 0,
                'match_ratio': 0,
                'rms_error': float('inf')
            })
    
    # Create a summary report
    summary_path = os.path.join(output_dir, "alignment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Alignment Summary for All Image Pairs\n")
        f.write("====================================\n\n")
        
        for result in alignment_results:
            f.write(f"Pair {result['pair_id']}:\n")
            f.write(f"  Success: {result['alignment_success']}\n")
            f.write(f"  Matching stars: {result['match_count']}\n")
            f.write(f"  Match ratio: {result['match_ratio']:.2f}\n")
            if result['rms_error'] != float('inf'):
                f.write(f"  RMS error: {result['rms_error']:.2f} pixels\n")
            else:
                f.write(f"  RMS error: N/A\n")
            f.write(f"  Notes: {result['reason']}\n\n")
    
    print("\nAll image sets processed.")
    print(f"Summary report saved to {summary_path}")

if __name__ == "__main__":
    # Path to the JSON file
    json_path = "kbo_image_sets.json"
    
    # Output directory
    output_dir = os.path.join("data", "processed_images")
    
    # Process the images
    align_and_visualize_star_matches(json_path, output_dir, downsample_factor=4)