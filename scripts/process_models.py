#!/usr/bin/env python3
"""
Process models from MONAI Model Zoo to create a model_data.json file for the website.
HuggingFace models are now managed in the model-zoo repository.
"""

import os
import json
from pathlib import Path
import markdown
from bs4 import BeautifulSoup

def get_latest_version(versions):
    """Find the latest version from a list of versioned model names."""
    numeric_versions = [v.split('_v')[-1] for v in versions if v.split('_v')[-1].replace('.', '').isdigit()]
    return max(numeric_versions, key=lambda k: tuple(map(int, k.split('.')))) if numeric_versions else max(versions)

def get_model_info(model_dir):
    """Extract metadata and README from a MONAI Model Zoo model."""
    metadata_path = os.path.join(model_dir, "configs", "metadata.json")
    readme_path = os.path.join(model_dir, "docs", "README.md")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(readme_path, 'r') as f:
        readme_content = f.read()
        readme_html = markdown.markdown(readme_content, extensions=['tables', 'fenced_code'])
        readme_soup = BeautifulSoup(readme_html, "html.parser")
    
    return metadata, readme_soup

def process_models():
    """Process all models and create a unified model_data.json file."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    models_dir = script_dir / "model-zoo" / "models"
    model_info_path = models_dir / "model_info.json"
    output_file = script_dir / "model_data.json"

    all_models = {}
    
    # Process regular model-zoo models if the model_info.json exists
    if model_info_path.exists():
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)

        for model_full_name, model_data in model_info.items():
            base_model_name = model_full_name.split('_v')[0]
            model_dir = models_dir / base_model_name
            
            if not model_dir.exists():
                continue

            versions = [k for k in model_info if k.startswith(base_model_name)]
            latest_version = get_latest_version(versions)
            
            try:
                metadata, readme_soup = get_model_info(str(model_dir))
                
                all_models[base_model_name] = {
                    "model_name": metadata.get("name", base_model_name.replace("_", " ").capitalize()),
                    "description": metadata.get("description", ""),
                    "authors": metadata.get("authors", "MONAI team"),
                    "papers": metadata.get("references", []),
                    "version": latest_version,
                    "model_id": base_model_name,
                    "readme": str(readme_soup),
                    "download_url": model_data.get("source", ""),
                    "changelog": metadata.get("changelog", {})
                }
                print(f"Processed model: {base_model_name}")
            except Exception as e:
                print(f"Error processing {base_model_name}: {str(e)}")
                continue

    # Write the combined data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_models, f, indent=4, ensure_ascii=False)
    
    print(f"Generated model data with {len(all_models)} models")
    return all_models

if __name__ == "__main__":
    process_models()