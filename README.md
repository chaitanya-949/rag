 # RAG Retrieval Augmented Generation

This project utilizes techniques from retrieval-augmented generation (RAG) to extract information from PDF documents and generate responses based on user queries and prompts.

## Description

The project includes the following key features:
## Data preparation
1. **PDF Text Extraction**: Extracts text from a PDF document, segmenting it into paragraphs for further analysis.
## retrival mechanism
2. **Contextual Similarity Search**: Searches for relevant passages in the PDF document based on contextual similarity to a given query using sentence embeddings.
## generation component
3. **Text Generation**: Utilizes pre-trained GPT-2 models for text generation, incorporating retrieved passages and user prompts.

## Install the required Python libraries:
pip install PyPDF2 sentence-transformers transformers torch


## Usage

1. **Extract Text from PDF**: Replace `pdf_path` variable with the path to your PDF document and run the script to extract text.

2. **Search with Similarity**: Provide a query and run the `search_with_similarity()` function to search for relevant passages.

3. **Generate Text**: Utilize the `generate_text()` and `generate_response()` functions to generate text based on retrieved passages and user prompts.

## Example

```python
# Example usage (assuming you have segmented text 'text')
query = "In case UMB is being offered to the animal for licking, how much quantity of urea treated straw should be fed to animal?"
relevant_passages = search_with_similarity(query, text)
generated_response = generate_response(relevant_passages)
print("Generated Response:", generated_response)

