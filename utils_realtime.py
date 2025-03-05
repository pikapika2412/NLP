# Configuration
API_KEY = "753d07b9-7030-4de2-af83-ed0c49e6f5a0"
MODEL_NAME = "llama3.1-8b"
FOLDER_DIR = 'D:/Phenikaa/Studying/NLP/Research-Poster-Understanding/dataset/images'
OUTPUT_FILE = "./poster_analysis_results.json"

import os
import json
import argparse
import time
from pathlib import Path
import pytesseract
from PIL import Image
import cv2
import numpy as np
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import re


def setup_client():
    """Initialize the OpenAI client to use LlamaAPI"""
    return OpenAI(
        api_key=API_KEY,
        base_url="https://api.llama-api.com"
    )
client = setup_client()

def load_existing_results():
    """Load existing analysis results if available"""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {OUTPUT_FILE}, starting fresh")
    return []

def save_results(results):
    """Save the analysis results to a JSON file"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def preprocess_image(image_path):
    """Preprocess the image for better OCR results"""
    # Load the image
    img = cv2.imread(str(image_path))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Invert the binary image back
    binary = cv2.bitwise_not(binary)
    
    return binary

def extract_text_from_image(image_path):
    """Extract text from an image using OCR"""
    try:
        # Preprocess the image
        preprocessed = preprocess_image(image_path)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(preprocessed)
        
        # Clean up the text
        text = text.replace('\n\n', ' [PARAGRAPH] ')
        text = text.replace('\n', ' ')
        text = text.replace('[PARAGRAPH]', '\n\n')
        text = text.strip()
        
        return text
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {str(e)}")
        return ""

def parse_structured_text(text):
    """Parse structured text into components with more flexible pattern matching"""
    # Initialize empty dictionary for results
    result = {
        "title": "",
        "abstract": "",
        "method": "",
        "result": ""
    }
    
    # Check for introductory text and remove it if present
    intro_pattern = r"^(Here are the extracted sections|I have analyzed the poster|The following sections contain).*?\n\n"
    text = re.sub(intro_pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # First try the markdown-style format with ** around section names
    md_pattern = r"\*\*SECTION: ([^*]+)\*\*\s*\n\s*CONTENT:(.*?)(?=\n\s*\*\*SECTION:|$)"
    md_matches = re.findall(md_pattern, text, re.DOTALL)
    
    if md_matches:
        for section_name, content in md_matches:
            section_type = section_name.strip().lower()
            content = content.strip()
            
            if section_type == "title":
                result["title"] = clean_text(content)
            elif section_type == "abstract":
                result["abstract"] = clean_text(content)
            elif section_type in ["method", "methods"]:
                result["method"] = clean_text(content)
            elif section_type in ["result", "results"]:
                result["result"] = clean_text(content)
        
        # If we found content using the markdown pattern, return the result
        if any(result.values()):
            return result
    
    # Next try the formal SECTION: format
    sections = text.split("SECTION: ")
    found_sections = False
    
    for section in sections:
        if not section.strip():
            continue
            
        # Find the section type and content
        parts = section.split("\nCONTENT: ", 1)
        if len(parts) == 2:
            found_sections = True
            section_type = parts[0].strip().lower()
            content = parts[1].strip()
            
            if section_type == "title":
                result["title"] = clean_text(content)
            elif section_type == "abstract":
                result["abstract"] = clean_text(content)
            elif section_type in ["method", "methods"]:
                result["method"] = clean_text(content)
            elif section_type in ["result", "results"]:
                result["result"] = clean_text(content)
    
    # If formal sections were found, return the result
    if found_sections:
        return result
    
    # If no sections were found, try alternative pattern matching
    # Common patterns for section headers
    patterns = {
        "title": [r'(?:^|\n)(?:\d+\.\s*)?TITLE\s*[:.-]*\s*(.*?)(?=\n\s*(?:ABSTRACT|INTRODUCTION|METHODS|RESULTS|$))',
                 r'(?:^|\n)(?:\d+\.\s*)?SECTION: TITLE\s*(.*?)(?=\n\s*(?:SECTION|ABSTRACT|INTRODUCTION|METHODS|RESULTS|$))'],
        "abstract": [r'(?:^|\n)(?:\d+\.\s*)?ABSTRACT\s*[:.-]*\s*(.*?)(?=\n\s*(?:METHODS|METHODOLOGY|APPROACH|RESULTS|$))',
                    r'(?:^|\n)(?:\d+\.\s*)?SECTION: ABSTRACT\s*(.*?)(?=\n\s*(?:SECTION|METHODS|METHODOLOGY|APPROACH|RESULTS|$))'],
        "method": [r'(?:^|\n)(?:\d+\.\s*)?(?:METHODS?|METHODOLOGY|APPROACH)\s*[:.-]*\s*(.*?)(?=\n\s*(?:RESULTS|FINDINGS|CONCLUSION|$))',
                  r'(?:^|\n)(?:\d+\.\s*)?SECTION: (?:METHODS?|METHODOLOGY)\s*(.*?)(?=\n\s*(?:SECTION|RESULTS|FINDINGS|CONCLUSION|$))'],
        "result": [r'(?:^|\n)(?:\d+\.\s*)?(?:RESULTS?|FINDINGS|OUTCOMES)\s*[:.-]*\s*(.*?)(?=\n\s*(?:DISCUSSION|CONCLUSION|REFERENCES|$))',
                 r'(?:^|\n)(?:\d+\.\s*)?SECTION: (?:RESULTS?|FINDINGS)\s*(.*?)(?=$)']
    }
    
    # Try to find each section using the patterns
    for section, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and match.group(1).strip():
                result[section] = clean_text(match.group(1).strip())
                break
    
    # If still no sections found, try numbered sections (1. Title, 2. Abstract, etc.)
    if all(not v for v in result.values()):
        numbered_sections = re.split(r'\n\s*\d+\.\s+', '\n' + text)
        
        # Check if we have at least 4 sections (title, abstract, methods, results)
        if len(numbered_sections) >= 5:  # First element is empty due to leading \n
            # Map the sections to the respective fields
            if len(numbered_sections[1].strip()) > 0:
                result["title"] = clean_text(numbered_sections[1].strip())
            if len(numbered_sections) > 2 and len(numbered_sections[2].strip()) > 0:
                result["abstract"] = clean_text(numbered_sections[2].strip())
            if len(numbered_sections) > 3 and len(numbered_sections[3].strip()) > 0:
                result["method"] = clean_text(numbered_sections[3].strip())
            if len(numbered_sections) > 4 and len(numbered_sections[4].strip()) > 0:
                result["result"] = clean_text(numbered_sections[4].strip())
    
    # Last resort: try to find content based on explicit section headers in the text
    if all(not v for v in result.values()):
        lines = text.split('\n')
        current_section = None
        section_content = {
            "title": [],
            "abstract": [],
            "method": [],
            "result": []
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a section header
            lower_line = line.lower()
            if "title" in lower_line and (len(line) < 30 or ":" in line):
                current_section = "title"
                # If the title is in the same line, extract it
                if ":" in line:
                    result["title"] = clean_text(line.split(":", 1)[1].strip())
                    current_section = None
                continue
            elif any(kw in lower_line for kw in ["abstract", "overview", "introduction"]) and (len(line) < 30 or ":" in line):
                current_section = "abstract"
                if ":" in line:
                    result["abstract"] = clean_text(line.split(":", 1)[1].strip())
                    current_section = None
                continue
            elif any(kw in lower_line for kw in ["method", "approach", "methodology"]) and (len(line) < 30 or ":" in line):
                current_section = "method"
                if ":" in line:
                    result["method"] = clean_text(line.split(":", 1)[1].strip())
                    current_section = None
                continue
            elif any(kw in lower_line for kw in ["result", "findings", "outcomes"]) and (len(line) < 30 or ":" in line):
                current_section = "result"
                if ":" in line:
                    result["result"] = clean_text(line.split(":", 1)[1].strip())
                    current_section = None
                continue
                
            # Add content to the current section
            if current_section:
                section_content[current_section].append(line)
        
        # Set content for sections that don't have content yet
        for section, content_lines in section_content.items():
            if not result[section] and content_lines:
                result[section] = clean_text(" ".join(content_lines))
    
    return result

def clean_text(text):
    """Clean up text by removing bullet points and other formatting"""
    # Replace bullet points like * or • at the beginning of lines
    text = re.sub(r'^\s*[\*\•\-\–\—]\s*', '', text, flags=re.MULTILINE)
    
    # Remove numbers followed by periods at the beginning of lines (like "1. ")
    text = re.sub(r'^\s*\d+[\.\)]\s*', '', text, flags=re.MULTILINE)
    
    # Remove common OCR noise patterns
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove references in the format [1], [2-5], etc.
    text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
    
    # Remove page numbers, often isolated digits
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove common poster section headers that we don't want as content
    for header in ["references", "acknowledgments", "acknowledgements", "contact", "funding", "future work"]:
        text = re.sub(fr'^\s*{header}\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text

def analyze_poster_text(text, client=client):
    """Send the extracted poster text to Llama API for analysis"""
    try:
        # First attempt with standard prompt
        parsed_content, token_usage = attempt_analysis(client, text, standard_prompt=True)
        
        # Check if any key field is empty
        missing_fields = [field for field, content in parsed_content.items() if not content]
        
        # If there are missing fields, retry with a focused prompt
        if missing_fields:
            print(f"Missing fields: {', '.join(missing_fields)}. Retrying with focused prompt...")
            
            # Retry with focused prompt for missing fields
            focused_content, additional_tokens = attempt_analysis(client, text, 
                                                               standard_prompt=False, 
                                                               missing_fields=missing_fields)
            
            # Update only the missing fields
            for field in missing_fields:
                if focused_content.get(field):
                    parsed_content[field] = focused_content[field]
            
            # Update token usage
            token_usage += additional_tokens
            
        return parsed_content, token_usage
            
    except Exception as e:
        print(f"Error analyzing poster text: {str(e)}")
        
        # Provide more detailed error information if available
        if hasattr(e, 'response'):
            try:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            except:
                pass
        return None, 0

def attempt_analysis(client, text, standard_prompt=True, missing_fields=None):
    """Attempt to analyze the poster text with a specific prompt"""
    
    if standard_prompt:
        # Standard analysis prompt with OCR noise handling instructions
        prompt = f"""The following text was extracted from an academic poster using OCR. 
        Please analyze it and extract the following sections.

        IMPORTANT: This is OCR output, so it contains noise and irrelevant text such as:
        - Conference names, dates, logos, affiliations
        - Author information and contact details
        - Page numbers, references, acknowledgments
        - Irrelevant headers/footers and navigational elements
        
        Focus only on extracting the core academic content for each section. Filter out this noise.

        For each section, follow this exact format:
        SECTION: [section name]
        CONTENT: [content for that section]

        Extract these sections:
        1. SECTION: TITLE
        2. SECTION: ABSTRACT (if no explicit abstract, use the overview/introduction section content)
        3. SECTION: METHODS
        4. SECTION: RESULTS

        Here is the text from the poster:
        {text}"""
        
        system_content = "You are an academic poster analyzer. Extract key information from poster text and organize it into clean, structured sections while filtering out OCR noise."
    else:
        # Create focused prompt for missing fields with OCR noise handling
        missing_sections = []
        instructions = []
        
        if "title" in missing_fields:
            missing_sections.append("TITLE")
            instructions.append("For the TITLE, extract only the main title of the poster. Ignore subtitles, author names, institutions, conference names.")
            
        if "abstract" in missing_fields:
            missing_sections.append("ABSTRACT")
            instructions.append("For the ABSTRACT, look for an overview or introduction section that summarizes the work. " 
                              "If no explicit abstract, use the first substantive paragraph or the introduction section content. "
                              "Exclude acknowledgments, references, and institutional information.")
            
        if "method" in missing_fields:
            missing_sections.append("METHODS")
            instructions.append("For the METHODS, identify sections describing approaches, techniques, algorithms, or procedures used in the research. "
                                "Focus on the core methodology and exclude references, figure labels, and supplementary details.")
            
        if "result" in missing_fields:
            missing_sections.append("RESULTS")
            instructions.append("For the RESULTS, locate findings, outcomes, performance metrics, or conclusions of the research. "
                               "Focus on the core findings and exclude references, acknowledgments, and future work.")
        
        sections_text = "\n".join([f"{i+1}. SECTION: {section}" for i, section in enumerate(missing_sections)])
        instructions_text = "\n".join(instructions)
        
        prompt = f"""I need your help identifying missing sections from this academic poster text extracted by OCR.
        
        IMPORTANT: OCR text contains noise like conference names, logos, author details, page numbers, etc.
        Filter out this irrelevant information and focus only on the core academic content.
        
        {instructions_text}
        
        For each section, follow this exact format:
        SECTION: [section name]
        CONTENT: [content for that section]
        
        Extract these sections:
        {sections_text}
        
        Here is the text from the poster:
        {text}"""
        
        system_content = "You are a specialized academic text analyzer focusing on finding specific sections that were missed in initial analysis while filtering out OCR noise."
    
    # Create and send the request using the OpenAI client
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    # Process the response
    response_content = response.choices[0].message.content.strip()
    
    # Parse the structured text into components
    parsed_content = parse_structured_text(response_content)
    #open a temp_output file to store the response
    with open('temp_output.txt', 'w') as f:
        f.write(response_content)
    
    token_usage = response.usage.total_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens') else 0
    return parsed_content, token_usage

def main():
    global MODEL_NAME
    parser = argparse.ArgumentParser(description='Extract and analyze academic posters using OCR and Llama API')
    parser.add_argument('--folder', type=str, default=FOLDER_DIR, help='Path to folder containing poster images')
    parser.add_argument('--count', type=int, default=1, help='Number of images to process (all if not specified)')
    args = parser.parse_args()
    
    # Check if the folder exists
    poster_folder = Path(args.folder)
    if not poster_folder.exists() or not poster_folder.is_dir():
        print(f"Error: {args.folder} is not a valid directory")
        return
    
    # Initialize client
    client = setup_client()
    results = load_existing_results()
        
    # Get processed file names
    processed_files = set(item["file_name"] for item in results)
    
    # Get the list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    poster_files = [f for f in poster_folder.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions 
                    and f.name not in processed_files]
    
    # Sort files to ensure consistent order
    poster_files.sort()
    
    # Limit the number of files to process if specified
    if args.count is not None:
        poster_files = poster_files[:args.count]
    
    print(f"Found {len(poster_files)} new poster images to process")
    
    # Create a progress bar
    progress_bar = tqdm(total=len(poster_files), desc="Processing posters", unit="image")
    
    for i, poster_file in enumerate(poster_files):
            
        progress_bar.set_postfix(file=poster_file.name, refresh=True)
        
        # Step 1: Extract text from image using OCR
        progress_bar.set_description(f"Extracting text from {poster_file.name}")
        extracted_text = extract_text_from_image(poster_file)
        
        if not extracted_text:
            progress_bar.write(f"Failed to extract text from {poster_file.name}, skipping...")
            progress_bar.update(1)
            continue
            
        # Step 2: Analyze the extracted text
        progress_bar.set_description(f"Analyzing text from {poster_file.name}")
        analysis_result, token_count = analyze_poster_text(client, extracted_text)
        
        # Handle completely failed analysis
        if analysis_result is None or all(not content for content in analysis_result.values()):
            progress_bar.write(f"Failed to analyze {poster_file.name}, skipping...")
            progress_bar.update(1)
            continue
        
        # Add to results if analysis was successful
        if analysis_result:
            result_entry = {
                "id": str(len(results) + 1),
                "file_name": poster_file.name,
                "annotation": analysis_result,
            }
            results.append(result_entry)
            
            # Save progress after each analysis
            save_results(results)
        
        # Add a small delay between requests to avoid rate limits
        time.sleep(1.5)
        
        # Update progress bar
        progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    processed_count = min(i+1, len(poster_files)) if poster_files else 0
    print(f"\nAnalysis complete. Processed {processed_count} out of {len(poster_files)} images.")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()