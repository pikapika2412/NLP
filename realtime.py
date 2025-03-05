import gradio as gr
import utils_realtime as utils
from PIL import Image


PROMPT = '''
    Focus only on extracting the core academic content
    For each section, follow this exact format:
        SECTION: [section name]
        CONTENT: [content for that section]

        Extract these sections:
        1. SECTION: TITLE
        2. SECTION: ABSTRACT (if no explicit abstract, use the overview/introduction section content)
        3. SECTION: METHODS
        4. SECTION: RESULTS

    Here is the poster:
'''

def analyze_image(image, prompt):
    """Analyze the poster image and return the analysis result"""
    
    
    # Step 1: Extract text from image using OCR
    extracted_text = utils.extract_text_from_image(image)
    
    if not extracted_text:
        return "Failed to extract text from the image.", "", "", "", ""
    
    # Step 2: Analyze the extracted text
    analysis_result, token_count = utils.analyze_poster_text(extracted_text)
    
    if analysis_result is None or all(not content for content in analysis_result.values()):
        return "Failed to analyze the image.", "", "", "", ""
    
    title = analysis_result.get("title", "")
    abstract = analysis_result.get("abstract", "")
    methods = analysis_result.get("method", "")
    results = analysis_result.get("result", "")
    
    return title, abstract, methods, results

def main():
    """Create a Gradio interface for the poster analysis"""
    interface = gr.Interface(
        fn=analyze_image,
        inputs=[
            gr.Image(type="filepath", label="Upload Poster Image"),
            gr.Textbox(lines=2, placeholder=f"{PROMPT}", label="Prompt")
        ],
        outputs=[
            gr.Textbox(label="Title"),
            gr.Textbox(label="Abstract"),
            gr.Textbox(label="Methods"),
            gr.Textbox(label="Results")
        ],
        title="Poster Analysis Tool",
        description="Upload an academic poster image."
    )
    
    interface.launch()

if __name__ == "__main__":
    main()