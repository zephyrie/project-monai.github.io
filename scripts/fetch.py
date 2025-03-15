import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import markdown

def get_latest_version(versions):
    numeric_versions = [v.split('_v')[-1] for v in versions if v.split('_v')[-1].replace('.', '').isdigit()]
    return max(numeric_versions, key=lambda k: tuple(map(int, k.split('.')))) if numeric_versions else max(versions)

def get_model_info(model_dir):
    metadata_path = os.path.join(model_dir, "configs", "metadata.json")
    readme_path = os.path.join(model_dir, "docs", "README.md")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(readme_path, 'r') as f:
        readme_content = f.read()
        readme_html = markdown.markdown(readme_content, extensions=['tables', 'fenced_code'])
        readme_soup = BeautifulSoup(readme_html, "html.parser")
    
    return metadata, readme_soup

def get_hf_model_info(model_dir):
    metadata_path = os.path.join(model_dir, "metadata.json")
    readme_path = os.path.join(model_dir, "README.md")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(readme_path, 'r') as f:
        readme_content = f.read()
        # Strip front matter if present
        if readme_content.startswith('---'):
            parts = readme_content.split('---', 2)
            if len(parts) >= 3:
                readme_content = parts[2]
        readme_html = markdown.markdown(readme_content, extensions=['tables', 'fenced_code'])
        readme_soup = BeautifulSoup(readme_html, "html.parser")
    
    return metadata, readme_soup

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    models_dir = script_dir / "model-zoo" / "models"
    model_info_path = models_dir / "model_info.json"
    hf_models_dir = repo_root / "hf"
    output_file = script_dir / "model_data.json"

    all_models = {}
    
    # Process regular model-zoo models if the model_info.json exists
    if model_info_path.exists():
        # Load model_info.json
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
                    "changelog": metadata.get("changelog")  # Get changelog directly from metadata
                }
            except Exception as e:
                print(f"Error processing {base_model_name}: {str(e)}")
                continue
    
    # Process HF models
    if hf_models_dir.exists():
        for model_dir in hf_models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_id = model_dir.name
            
            try:
                metadata, readme_soup = get_hf_model_info(str(model_dir))
                
                # Extract paper references from metadata
                papers = []
                if "references" in metadata:
                    for ref in metadata["references"]:
                        if isinstance(ref, dict):
                            # Format citation string from structured reference
                            citation = f"{ref.get('title', '')}. "
                            if ref.get('authors'):
                                citation += f"{ref.get('authors', '')}. "
                            if ref.get('journal'):
                                citation += f"{ref.get('journal', '')}. "
                            if ref.get('year'):
                                citation += f"{ref.get('year', '')}. "
                            if ref.get('url'):
                                citation += f"[{ref.get('url', '')}]"
                            papers.append(citation)
                        else:
                            papers.append(ref)
                
                # Use model_url for download if available, otherwise construct a fallback URL
                download_url = metadata.get("model_url", "")
                
                all_models[f"hf_{model_id}"] = {
                    "model_name": metadata.get("name", model_id.capitalize()),
                    "description": metadata.get("description", ""),
                    "authors": metadata.get("authors", ""),
                    "papers": papers,
                    "version": metadata.get("version", "1.0.0"),
                    "model_id": f"hf_{model_id}",
                    "readme": str(readme_soup),
                    "download_url": download_url,
                    "changelog": metadata.get("changelog", {})
                }
                
                print(f"Processed HF model: {model_id}")
            except Exception as e:
                print(f"Error processing HF model {model_id}: {str(e)}")
                continue

    # Write the combined data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_models, f, indent=4, ensure_ascii=False)
    
    print(f"Generated model data with {len(all_models)} models")

if __name__ == "__main__":
    main()