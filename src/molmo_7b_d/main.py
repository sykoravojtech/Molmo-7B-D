from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

DEFAULT_CLASSES: List[str] = [
    "__background__", "text", "junction", "crossover", "terminal", "gnd", "vss",
    "voltage.dc", "voltage.ac", "voltage.battery", "resistor", "resistor.adjustable",
    "resistor.photo", "capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable",
    "inductor", "inductor.ferrite", "inductor.coupled", "transformer", "diode",
    "diode.light_emitting", "diode.thyrector", "diode.zener", "diac", "triac",
    "thyristor", "varistor", "transistor.bjt", "transistor.fet", "transistor.photo",
    "operational_amplifier", "operational_amplifier.schmitt_trigger", "optocoupler",
    "integrated_circuit", "integrated_circuit.ne555", "integrated_circuit.voltage_regulator",
    "xor", "and", "or", "not", "nand", "nor", "probe", "probe.current",
    "probe.voltage", "switch", "relay", "socket", "fuse", "speaker", "motor",
    "lamp", "microphone", "antenna", "crystal", "mechanical", "magnetic",
    "optical", "block", "unknown"
]

def load_model(model_id: str) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Molmo model."""
    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    return (
        AutoModelForCausalLM.from_pretrained(model_id, **kw),
        AutoProcessor.from_pretrained(model_id, **kw),
    )
    
def make_prompt(classes: List[str]) -> str:
    """Craft detection prompt."""
    # Create numbered list of classes for the model
    class_list = "\n".join([f"{i}: {name}" for i, name in enumerate(classes)])
    return (
        f"You are an expert object detection model specialized in analyzing electric circuit diagram symbols. "
        f"Your task is to detect every instance of the following symbol classes in the given circuit diagram:\n\n"
        f"{class_list}\n\n"
        f"For each detected symbol, return a JSON array where each element follows this exact format:\n"
        '{"class": <integer_class_index>, "bbox": [x_min, y_min, x_max, y_max], "score": <confidence_score>}\n\n'
        f"The output will be parsed into a dictionary with keys: 'boxes', 'labels', 'scores'.\n\n"
        f"Important requirements:\n"
        f"- Use integer class indices (0-{len(classes)-1}) corresponding to the numbered list above\n"
        f"- Bounding boxes MUST use format [x_min, y_min, x_max, y_max] with pixel coordinates\n"
        f"- (x_min, y_min) is the top-left corner, (x_max, y_max) is the bottom-right corner\n"
        f"- Ensure x_min < x_max and y_min < y_max\n"
        f"- Confidence scores should be between 0.0 and 1.0\n"
        f"- Output ONLY the JSON array, nothing else\n"
        f"- Detect ALL visible symbols, even if they appear multiple times or are partially visible\n"
        f"- You MUST find at least some symbols - empty results are not acceptable"
    )
    
def infer_to_raw(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
) -> str:
    """Run Molmo and return raw generated string."""
    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # Convert image inputs to the same dtype as the model
    if 'images' in inputs:
        inputs['images'] = inputs['images'].to(dtype=torch.bfloat16)
    
    with torch.autocast("cuda", torch.bfloat16):
        out = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=256, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )
    generated_tokens = out[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

def parse_predictions(raw: str, classes: List[str]) -> Dict[str, torch.Tensor]:
    """Parse raw JSON predictions into tensors."""
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON output: {e}")
        print(f"Raw output was: {raw}")
        raise ValueError(f"Failed to parse JSON output: {e}")
    
    # Handle empty predictions - return failure instead of empty tensors
    if not items:
        raise ValueError("No predictions found - model returned empty results")
    
    boxes = torch.tensor([item["bbox"] for item in items], dtype=torch.float32)
    scores = torch.tensor([item.get("score", 1.0) for item in items], dtype=torch.float32)
    
    # Handle integer indices only
    labels = []
    for item in items:
        class_val = item["class"]
        if isinstance(class_val, int) and 0 <= class_val < len(classes):
            labels.append(class_val)
        else:
            # Invalid index, default to background
            labels.append(0)
    
    labels = torch.tensor(labels, dtype=torch.int64)
    return {"boxes": boxes, "labels": labels, "scores": scores}

def main() -> None:
    """CLI entry-point."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "image",
        default="/storage/brno2/home/nademvit/vthesis/cg25/data/rpi_pico_sample/drafter_31/images/48c9152c-0-1.jpg",
        help="Path to the image file to analyze",
        nargs="?"  # Makes the argument optional, using default if not provided
    )
    p.add_argument("--model", default="allenai/Molmo-7B-D-0924")
    p.add_argument(
        "--classes", nargs='+', default=DEFAULT_CLASSES,
        help="List of symbol classes to detect."
    )
    args = p.parse_args()

    model, proc = load_model(args.model)
    img = Image.open(args.image).convert("RGB")
    prompt = make_prompt(args.classes)
    raw = infer_to_raw(model, proc, img, prompt)
    print("Raw output:")
    print(raw)
    
    try:
        pred = parse_predictions(raw, args.classes)
        print("Parsed predictions:")
        print(pred)
        
        # Save predictions to JSON file
        image_path = Path(args.image)
        image_name = image_path.stem  # Get filename without extension
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)  # Create outputs directory if it doesn't exist
        
        output_file = output_dir / f"{image_name}.json"
        
        # Convert tensors to lists for JSON serialization
        pred_serializable = {
            "boxes": pred["boxes"].tolist(),
            "labels": pred["labels"].tolist(), 
            "scores": pred["scores"].tolist()
        }
        
        with open(output_file, 'w') as f:
            json.dump(pred_serializable, f, indent=2)
        
        print(f"Predictions saved to: {output_file}")
        
    except ValueError as e:
        print(f"Failed to parse predictions: {e}")
        return

if __name__ == "__main__":
    main()
