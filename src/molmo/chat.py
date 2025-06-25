from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def load_model(model_id: str) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load Molmo model."""
    kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    return (
        AutoModelForCausalLM.from_pretrained(model_id, **kw),
        AutoProcessor.from_pretrained(model_id, **kw),
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
    if "images" in inputs:
        inputs["images"] = inputs["images"].to(dtype=torch.bfloat16)

    with torch.autocast("cuda", torch.bfloat16):
        out = model.generate_from_batch(
            inputs,
            GenerationConfig(
                max_new_tokens=1024, stop_strings="<|endoftext|>", do_sample=False
            ),
            tokenizer=processor.tokenizer,
        )
    generated_tokens = out[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    return generated_text


def chat_loop(
    model: AutoModelForCausalLM, processor: AutoProcessor, image: Image.Image
) -> None:
    """Interactive chat loop with the model."""
    print("=" * 60)
    print("Molmo Chat Interface")
    print("=" * 60)
    print(f"Image loaded successfully!")
    print(
        "Type your questions or commands. Type 'quit', 'exit', or 'q' to end the session."
    )
    print("Type 'help' for usage tips.")
    print("-" * 60)

    while True:
        try:
            # Get user input
            print("---")
            user_input = input("Human_Prompt: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Handle empty input
            if not user_input:
                print("Please enter a question or command.")
                continue

            # Handle help command
            if user_input.lower() == "help":
                print("\nUsage tips:")
                print("- Ask questions about the image content")
                print("- Request analysis of specific elements")
                print("- Ask for descriptions or explanations")
                print("- Example: 'What components do you see in this circuit?'")
                print("- Example: 'How many resistors are in the image?'")
                print("- Example: 'Describe the layout of this circuit diagram'")
                continue

            # Show that we're processing
            print("===")
            print("Model_Response: ", end="", flush=True)

            # Get model response
            try:
                response = infer_to_raw(model, processor, image, user_input)
                print(response)
            except Exception as e:
                print(f"Error generating response: {e}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nSession ended. Goodbye!")
            break


def get_vlm_prompt() -> str:
    """Return VLM prompt for circuit-diagram symbol detection."""
    return """You are a specialised circuit-diagram interpreter. Given the image of an electrical circuit diagram, identify which symbol classes from the list below appear in the image. List only the classes you find, in a python list, and nothing else. Symbol classes: ["terminal", "gnd", "vss", "voltage.dc", "voltage.ac", "voltage.battery", "resistor", "resistor.adjustable", "resistor.photo", "capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable", "inductor", "inductor.ferrite", "inductor.coupled", "transformer", "diode", "diode.light_emitting", "diode.thyrector", "diode.zener", "diac", "triac", "thyristor", "varistor", "transistor.bjt", "transistor.fet", "transistor.photo", "operational_amplifier", "operational_amplifier.schmitt_trigger", "optocoupler", "integrated_circuit", "integrated_circuit.ne555", "integrated_circuit.voltage_regulator", "xor", "and", "or", "not", "nand", "nor", "probe", "probe.current", "probe.voltage", "switch", "relay", "socket", "fuse", "speaker", "motor", "lamp", "microphone", "antenna", "crystal", "mechanical", "magnetic", "optical", "block"]"""


def get_vlm_resistor_xml_prompt(cls_name: str) -> str:
    """Return VLM prompt for zero-shot resistor detection in XML format."""
    return f"""You are a specialised circuit-diagram interpreter performing zero-shot inference. Given the image of an electrical circuit diagram, identify all symbols of class "{cls_name}". For each symbol from that class, output one <points> tag and nothing else, in the exact format: <points x1="…" y1="…" x2="…" y2="…" alt="{cls_name}">{cls_name}</points> Use the coordinates of the bounding polygon of each symbol. Do not include any additional text or tags. Example: <points x1="20.8" y1="42.4" x2="21.0" y2="53.6" alt="{cls_name}">{cls_name}</points>"""


def main() -> None:
    """CLI entry-point for chat interface."""
    p = argparse.ArgumentParser(description="Interactive chat with Molmo model")
    p.add_argument(
        "--image",
        default="data/images/48c9152c-0-4.jpg",
        help="Path to the image file to analyze",
        nargs="?",  # Makes the argument optional, using default if not provided
    )
    p.add_argument("--model", default="allenai/Molmo-7B-D-0924", help="Model ID to use")

    args = p.parse_args()

    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    print("Loading model... This may take a moment.")
    try:
        model, processor = load_model(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Loading image: {args.image}")
    try:
        image = Image.open(args.image).convert("RGB")
        print(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Start chat loop
    chat_loop(model, processor, image)


if __name__ == "__main__":
    main()
