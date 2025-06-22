"""
Visualization tool for comparing Molmo predictions with ground truth annotations.
"""
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse


def parse_xml_annotations(xml_path: str) -> Dict[str, List[Dict]]:
    """Parse XML annotation file and extract bounding boxes by object type."""
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
    """Parse Molmo model response and extract point coordinates."""
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


def get_component_colors():
    """Define colors for different component types."""
    return {
        'resistor': '#FF6B6B',     # Red
        'junction': '#4ECDC4',     # Teal
        'switch': '#45B7D1',       # Blue
        'terminal': '#96CEB4',     # Green
        'capacitor': '#FECA57',    # Yellow
        'inductor': '#FF9FF3',     # Pink
        'diode': '#54A0FF',        # Light Blue
        'transistor': '#5F27CD',   # Purple
        'text': '#D1D1D1'          # Light Gray (for text objects)
    }


def visualize_predictions(image_path: str, xml_path: str, molmo_response: str, output_path: str):
    """
    Create visualization comparing ground truth annotations with Molmo predictions.
    
    Args:
        image_path: Path to the circuit image
        xml_path: Path to XML annotation file
        molmo_response: Raw Molmo response containing <points> tags
        output_path: Path to save the visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_width, image_height = image.size
    
    # Parse annotations and predictions
    gt_annotations = parse_xml_annotations(xml_path)
    molmo_predictions = parse_molmo_points(molmo_response)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Color scheme
    colors = get_component_colors()
    
    # Plot 1: Ground Truth Only (excluding text)
    ax1.imshow(image)
    ax1.set_title('Ground Truth Annotations\n(excluding text objects)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    gt_legend_elements = []
    for obj_type, bboxes in gt_annotations.items():
        if obj_type == 'text':  # Skip text objects
            continue
            
        color = colors.get(obj_type, '#888888')
        for bbox in bboxes:
            rect = patches.Rectangle(
                (bbox['xmin'], bbox['ymin']),
                bbox['xmax'] - bbox['xmin'],
                bbox['ymax'] - bbox['ymin'],
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.8
            )
            ax1.add_patch(rect)
        
        # Add to legend (only once per type)
        if obj_type not in [elem.get_label() for elem in gt_legend_elements]:
            gt_legend_elements.append(
                patches.Patch(color=color, label=f'{obj_type} ({len(bboxes)})')
            )
    
    ax1.legend(handles=gt_legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Plot 2: Molmo Predictions Only
    ax2.imshow(image)
    ax2.set_title('Molmo Predictions', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    molmo_legend_elements = []
    for component_type, coordinates in molmo_predictions:
        # Convert coordinates from percentage to absolute pixels
        abs_coords = [(x * image_width / 100.0, y * image_height / 100.0) for x, y in coordinates]
        
        # Map plural to singular for consistent coloring
        singular_type = component_type.rstrip('s') if component_type.endswith('s') else component_type
        color = colors.get(singular_type, '#888888')
        
        # Plot points
        for x, y in abs_coords:
            ax2.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)
        
        # Add to legend
        molmo_legend_elements.append(
            patches.Patch(color=color, label=f'{component_type} ({len(coordinates)})')
        )
    
    ax2.legend(handles=molmo_legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Plot 3: Combined View
    ax3.imshow(image)
    ax3.set_title('Combined: Ground Truth + Molmo Predictions', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Draw ground truth bounding boxes (excluding text)
    for obj_type, bboxes in gt_annotations.items():
        if obj_type == 'text':  # Skip text objects
            continue
            
        color = colors.get(obj_type, '#888888')
        for bbox in bboxes:
            rect = patches.Rectangle(
                (bbox['xmin'], bbox['ymin']),
                bbox['xmax'] - bbox['xmin'],
                bbox['ymax'] - bbox['ymin'],
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.6,
                linestyle='-'
            )
            ax3.add_patch(rect)
    
    # Draw Molmo predictions
    for component_type, coordinates in molmo_predictions:
        # Convert coordinates from percentage to absolute pixels
        abs_coords = [(x * image_width / 100.0, y * image_height / 100.0) for x, y in coordinates]
        
        # Map plural to singular for consistent coloring
        singular_type = component_type.rstrip('s') if component_type.endswith('s') else component_type
        color = colors.get(singular_type, '#888888')
        
        # Plot points
        for x, y in abs_coords:
            ax3.plot(x, y, 'x', color=color, markersize=10, markeredgewidth=3)
    
    # Combined legend
    combined_legend = []
    # Add GT legend items
    for obj_type, bboxes in gt_annotations.items():
        if obj_type == 'text':
            continue
        color = colors.get(obj_type, '#888888')
        combined_legend.append(
            patches.Patch(color=color, label=f'GT {obj_type} ({len(bboxes)}) □')
        )
    
    # Add Molmo legend items
    for component_type, coordinates in molmo_predictions:
        singular_type = component_type.rstrip('s') if component_type.endswith('s') else component_type
        color = colors.get(singular_type, '#888888')
        combined_legend.append(
            patches.Patch(color=color, label=f'Molmo {component_type} ({len(coordinates)}) ✕')
        )
    
    ax3.legend(handles=combined_legend, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add summary text
    total_gt = sum(len(bboxes) for obj_type, bboxes in gt_annotations.items() if obj_type != 'text')
    total_molmo = sum(len(coords) for _, coords in molmo_predictions)
    
    fig.suptitle(f'Circuit Analysis Comparison\n'
                f'Image: {Path(image_path).name} | '
                f'GT Objects: {total_gt} | '
                f'Molmo Predictions: {total_molmo}',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Print summary
    print("\nSUMMARY:")
    print("=" * 50)
    print("Ground Truth (excluding text):")
    for obj_type, bboxes in gt_annotations.items():
        if obj_type != 'text':
            print(f"  {obj_type}: {len(bboxes)}")
    
    print("\nMolmo Predictions:")
    for component_type, coordinates in molmo_predictions:
        print(f"  {component_type}: {len(coordinates)}")


def main():
    """CLI entry point for visualization tool."""
    parser = argparse.ArgumentParser(description="Visualize Molmo predictions vs ground truth")
    parser.add_argument("--image", required=True, help="Path to circuit image")
    parser.add_argument("--xml", required=True, help="Path to XML annotation file")
    parser.add_argument("--response", required=True, help="Molmo response text (quoted string)")
    parser.add_argument("--output", required=True, help="Output path for visualization")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not Path(args.xml).exists():
        print(f"Error: XML file not found: {args.xml}")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_predictions(args.image, args.xml, args.response, args.output)


if __name__ == "__main__":
    main()
