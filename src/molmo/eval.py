"""
Utility functions for evaluating Molmo model predictions against ground truth annotations.
"""
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path


def parse_xml_annotations(xml_path: str) -> Dict[str, List[Dict]]:
    """
    Parse XML annotation file and extract bounding boxes by object type.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        Dictionary mapping object types to list of bounding boxes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = {}
    
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        bbox_dict = {
            'xmin': int(bbox.find('xmin').text),
            'xmax': int(bbox.find('xmax').text),
            'ymin': int(bbox.find('ymin').text),
            'ymax': int(bbox.find('ymax').text)
        }
        
        if obj_name not in annotations:
            annotations[obj_name] = []
        annotations[obj_name].append(bbox_dict)
    
    return annotations


def parse_molmo_points(response_text: str) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """
    Parse Molmo model response and extract point coordinates.
    
    Args:
        response_text: Raw response text from Molmo model
        
    Returns:
        List of tuples containing (component_type, list_of_coordinates)
    """
    # Find all <points> tags
    point_pattern = r'<points\s+([^>]+)\s+alt="([^"]+)">([^<]+)</points>'
    matches = re.findall(point_pattern, response_text)
    
    results = []
    
    for attributes, component_type, content in matches:
        # Extract coordinates from attributes
        coord_pattern = r'x(\d+)="([^"]+)"\s+y\1="([^"]+)"'
        coord_matches = re.findall(coord_pattern, attributes)
        
        coordinates = []
        for _, x, y in coord_matches:
            coordinates.append((float(x), float(y)))
        
        results.append((component_type, coordinates))
    
    return results


def point_in_bbox(x: float, y: float, bbox: Dict) -> bool:
    """
    Check if a point is inside a bounding box.
    
    Args:
        x: X coordinate of point
        y: Y coordinate of point
        bbox: Bounding box dictionary with xmin, xmax, ymin, ymax
        
    Returns:
        True if point is inside bbox, False otherwise
    """
    return (bbox['xmin'] <= x <= bbox['xmax'] and 
            bbox['ymin'] <= y <= bbox['ymax'])


def evaluate_molmo_predictions(molmo_response: str, xml_path: str, image_width: int, image_height: int) -> Dict:
    """
    Evaluate Molmo predictions against ground truth annotations.
    
    Args:
        molmo_response: Raw response text from Molmo model
        xml_path: Path to XML annotation file
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        Dictionary containing evaluation results
    """
    # Parse annotations and predictions
    gt_annotations = parse_xml_annotations(xml_path)
    molmo_predictions = parse_molmo_points(molmo_response)
    
    # Component type mapping (plural to singular)
    component_mapping = {
        'resistors': 'resistor',
        'junctions': 'junction', 
        'switches': 'switch',
        'terminals': 'terminal',
        'capacitors': 'capacitor',
        'inductors': 'inductor',
        'diodes': 'diode',
        'transistors': 'transistor'
    }
    
    results = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'by_component': {},
        'unmatched_predictions': [],
        'accuracy': 0.0
    }
    
    for component_type, coordinates in molmo_predictions:
        component_results = {
            'total': len(coordinates),
            'correct': 0,
            'incorrect_coords': []
        }
        
        # Convert coordinates to absolute pixels - Molmo appears to use percentage coordinates
        processed_coords = []
        for x, y in coordinates:
            # Convert from percentage to absolute pixels
            abs_x = x * image_width / 100.0
            abs_y = y * image_height / 100.0
            processed_coords.append((abs_x, abs_y))
        
        # Map component type (handle plural forms)
        gt_component_type = component_mapping.get(component_type, component_type)
        
        # Check if we have ground truth for this component type
        if gt_component_type in gt_annotations:
            gt_bboxes = gt_annotations[gt_component_type]
            
            for x, y in processed_coords:
                # Check if point falls in any ground truth bbox
                found_match = False
                for bbox in gt_bboxes:
                    if point_in_bbox(x, y, bbox):
                        component_results['correct'] += 1
                        results['correct_predictions'] += 1
                        found_match = True
                        break
                
                if not found_match:
                    component_results['incorrect_coords'].append((x, y))
                
                results['total_predictions'] += 1
        else:
            # No ground truth for this component type
            component_results['incorrect_coords'] = processed_coords
            results['total_predictions'] += len(coordinates)
            results['unmatched_predictions'].extend([(component_type, coord) for coord in processed_coords])
        
        component_results['accuracy'] = (component_results['correct'] / component_results['total'] 
                                       if component_results['total'] > 0 else 0.0)
        results['by_component'][component_type] = component_results
    
    # Calculate overall accuracy
    results['accuracy'] = (results['correct_predictions'] / results['total_predictions'] 
                          if results['total_predictions'] > 0 else 0.0)
    
    return results


def print_evaluation_report(results: Dict, debug: bool = False) -> None:
    """Print a formatted evaluation report."""
    print("=" * 60)
    print("MOLMO EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy: {results['accuracy']:.2%}")
    print(f"Total Predictions: {results['total_predictions']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Incorrect Predictions: {results['total_predictions'] - results['correct_predictions']}")
    print()
    
    if debug:
        print("DEBUG INFO:")
        print("-" * 40)
        # This will be filled in by the calling function if needed
        print()
    
    print("By Component Type:")
    print("-" * 40)
    for component_type, comp_results in results['by_component'].items():
        print(f"{component_type}:")
        print(f"  Total: {comp_results['total']}")
        print(f"  Correct: {comp_results['correct']}")
        print(f"  Accuracy: {comp_results['accuracy']:.2%}")
        if comp_results['incorrect_coords']:
            print(f"  Incorrect coordinates: {comp_results['incorrect_coords'][:3]}{'...' if len(comp_results['incorrect_coords']) > 3 else ''}")
        print()
    
    if results['unmatched_predictions']:
        print("Unmatched Predictions (no ground truth):")
        print("-" * 40)
        for component_type, coord in results['unmatched_predictions'][:5]:
            print(f"  {component_type}: {coord}")
        if len(results['unmatched_predictions']) > 5:
            print(f"  ... and {len(results['unmatched_predictions']) - 5} more")


def debug_coordinate_comparison(molmo_response: str, xml_path: str, image_width: int, image_height: int) -> None:
    """Debug function to show coordinate comparisons."""
    gt_annotations = parse_xml_annotations(xml_path)
    molmo_predictions = parse_molmo_points(molmo_response)
    
    print("DEBUG: COORDINATE COMPARISON")
    print("=" * 50)
    
    component_mapping = {
        'resistors': 'resistor',
        'junctions': 'junction', 
        'switches': 'switch',
        'terminals': 'terminal',
    }
    
    for component_type, coordinates in molmo_predictions:
        gt_component_type = component_mapping.get(component_type, component_type)
        print(f"\n{component_type} -> {gt_component_type}:")
        print(f"Molmo coordinates (raw): {coordinates}")
        
        # Convert to absolute coordinates
        abs_coords = [(x * image_width / 100.0, y * image_height / 100.0) for x, y in coordinates]
        print(f"Molmo coordinates (abs): {abs_coords}")
        
        if gt_component_type in gt_annotations:
            print(f"Ground truth bboxes:")
            for i, bbox in enumerate(gt_annotations[gt_component_type]):
                print(f"  {i+1}: {bbox}")
                # Check which points fall in this bbox
                for j, (x, y) in enumerate(abs_coords):
                    if point_in_bbox(x, y, bbox):
                        print(f"    Point {j+1} ({x:.1f}, {y:.1f}) is INSIDE this bbox")
        else:
            print(f"No ground truth found for {gt_component_type}")
    print("=" * 50)


def visualize_predictions(image_path: str, molmo_response: str, xml_path: str) -> None:
    """
    Visualize Molmo predictions and ground truth annotations on the image.
    
    Args:
        image_path: Path to the image file
        molmo_response: Raw response text from Molmo model
        xml_path: Path to XML annotation file
    """
    # Parse annotations and predictions
    gt_annotations = parse_xml_annotations(xml_path)
    molmo_predictions = parse_molmo_points(molmo_response)
    
    # Load the image
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Colors for different component types
    colors = {
        'resistor': 'red',
        'junction': 'blue',
        'switch': 'green',
        'terminal': 'purple',
        'capacitor': 'orange',
        'inductor': 'cyan',
        'diode': 'magenta',
        'transistor': 'yellow'
    }
    
    # Plot ground truth boxes
    for obj_name, bboxes in gt_annotations.items():
        for bbox in bboxes:
            xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']
            plt.gca().add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                                   linewidth=2, edgecolor='black', facecolor='none'))
            plt.text(xmin, ymin, obj_name, fontsize=12, color='black', verticalalignment='top')
    
    # Plot Molmo predicted points
    for component_type, coordinates in molmo_predictions:
        color = colors.get(component_type, 'white')
        for (x, y) in coordinates:
            plt.scatter(x, y, color=color, s=100, edgecolor='black', label=f"Predicted {component_type}")
    
    plt.title("Molmo Predictions and Ground Truth Annotations")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()


print("=" * 50)


def visualize_predictions_to_file(image_path: str, molmo_response: str, xml_path: str, output_path: str = None) -> str:
    """
    Visualize Molmo predictions and ground truth annotations on the image and save to file.
    
    Args:
        image_path: Path to the image file
        molmo_response: Raw response text from Molmo model
        xml_path: Path to XML annotation file
        output_path: Path to save the visualization (if None, will auto-generate)
        
    Returns:
        Path to the saved visualization
    """
    # Parse annotations and predictions
    gt_annotations = parse_xml_annotations(xml_path)
    molmo_predictions = parse_molmo_points(molmo_response)
    
    # Load the image
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    # Create figure
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    
    # Colors for different component types
    colors = {
        'resistor': 'red',
        'junction': 'blue', 
        'switch': 'green',
        'terminal': 'purple',
        'capacitor': 'orange',
        'inductor': 'cyan',
        'diode': 'magenta',
        'transistor': 'yellow'
    }
    
    # Component type mapping for Molmo predictions
    component_mapping = {
        'resistors': 'resistor',
        'junctions': 'junction', 
        'switches': 'switch',
        'terminals': 'terminal',
        'capacitors': 'capacitor',
        'inductors': 'inductor',
        'diodes': 'diode',
        'transistors': 'transistor'
    }
    
    # Plot ground truth boxes (excluding text objects)
    bbox_count = 0
    for obj_name, bboxes in gt_annotations.items():
        if obj_name == 'text':  # Skip text annotations
            continue
            
        color = colors.get(obj_name, 'black')
        for i, bbox in enumerate(bboxes):
            xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']
            
            # Draw bounding box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                   linewidth=2, edgecolor=color, facecolor='none', 
                                   linestyle='-', alpha=0.8)
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(xmin, ymin - 5, f'GT-{obj_name}', fontsize=8, color=color, 
                    verticalalignment='bottom', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            bbox_count += 1
    
    # Plot Molmo predicted points
    point_count = 0
    used_colors = set()
    
    for component_type, coordinates in molmo_predictions:
        # Map to ground truth component type
        gt_component_type = component_mapping.get(component_type, component_type)
        color = colors.get(gt_component_type, 'white')
        
        # Convert coordinates to absolute pixels (assuming percentage coordinates)
        for j, (x, y) in enumerate(coordinates):
            abs_x = x * image_width / 100.0
            abs_y = y * image_height / 100.0
            
            # Plot point
            plt.scatter(abs_x, abs_y, color=color, s=120, edgecolor='black', 
                       linewidth=2, marker='o', alpha=0.9, zorder=10)
            
            # Add point number
            plt.text(abs_x + 5, abs_y + 5, f'{j+1}', fontsize=8, color='black',
                    weight='bold', 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
            point_count += 1
        
        used_colors.add((component_type, color))
    
    # Create legend
    legend_elements = []
    
    # Add ground truth legend entries
    for obj_name in gt_annotations.keys():
        if obj_name == 'text':
            continue
        color = colors.get(obj_name, 'black')
        legend_elements.append(patches.Patch(color=color, label=f'GT {obj_name} ({len(gt_annotations[obj_name])})'))
    
    # Add prediction legend entries
    for component_type, color in used_colors:
        coord_count = len([coord for ct, coords in molmo_predictions if ct == component_type for coord in coords])
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8, 
                                        markeredgecolor='black', markeredgewidth=1,
                                        label=f'Pred {component_type} ({coord_count})'))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Set title and labels
    plt.title(f"Ground Truth vs Molmo Predictions\nGT Boxes: {bbox_count}, Predicted Points: {point_count}", 
              fontsize=14, weight='bold')
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    
    # Tight layout to prevent legend cutoff
    plt.tight_layout()
    
    # Generate output path if not provided
    if output_path is None:
        image_name = Path(image_path).stem
        output_dir = Path("/storage/brno2/home/nademvit/vthesis/Molmo-7B-D/output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{image_name}_visualization.png"
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"Visualization saved to: {output_path}")
    return str(output_path)

# Image dimensions from the XML file
# Extract image dimensions from the XML file
def get_image_dimensions_from_xml(xml_path: str) -> Tuple[int, int]:
    """
    Extract image width and height from XML annotation file.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        Tuple containing (width, height)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size_elem = root.find('size')
    if size_elem is not None:
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        return width, height
    return None, None

if __name__ == "__main__":
    import os
    # Example usage
    # example_molmo_response = '''
    # Counting the <points x1="21.0" y1="76.5" x2="69.5" y2="37.9" x3="69.5" y3="51.5" x4="69.5" y4="65.5" x5="69.5" y5="78.0" x6="69.5" y6="92.0" alt="resistors">resistors</points> shows a total of 6.
    # '''
    example_molmo_response = '''
    Model_Response:  <points x1="20.8" y1="42.4" x2="21.0" y2="53.6" x3="70.5" y3="92.9" x4="70.8" y4="80.0" x5="71.0" y5="65.0" x6="71.1" y6="38.4" x7="71.1" y7="51.5" x8="81.8" y8="37.3" x9="81.8" y9="51.5" x10="81.8" y10="65.0" x11="81.8" y11="79.2" alt="terminal">terminal</points>
    Model_Response:  <points x1="20.8" y1="42.4" x2="21.0" y2="53.6" x3="69.0" y3="37.9" x4="69.0" y4="51.5" x5="69.0" y5="65.4" x6="69.0" y6="78.6" x7="69.0" y7="92.1" alt="resistor">resistor</points>
    Model_Response:  <points x1="20.8" y1="42.4" x2="21.0" y2="53.6" alt="switch">switch</points>
    '''
    
    image_path = "data/images/48c9152c-0-4.jpg"
    xml_path = f"data/annotations/{os.path.basename(image_path)[:-4]}.xml"
    
    # Get dimensions from XML
    image_width, image_height = get_image_dimensions_from_xml(xml_path)
    
    results = evaluate_molmo_predictions(example_molmo_response, xml_path, image_width, image_height)
    print_evaluation_report(results)
    visualize_predictions(image_path, example_molmo_response, xml_path)
    visualize_predictions_to_file(image_path, example_molmo_response, xml_path)
