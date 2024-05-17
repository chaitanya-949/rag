import streamlit as st
import PyPDF2
import re
import warnings
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

warnings.filterwarnings('ignore')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
            text += "\n\n"  # Add two newline characters to separate pages

    # Add delimiter before each section title
    text = text.replace("\n\n", "\n     \n\n")  # Replace double newline characters with a delimiter

    # Add new paragraph after the text "Question"
    text = text.replace("Question", "\nQuestion\n")

    return text

# Function to segment text into paragraphs
def segment_text(text):
    pattern = r"\n{2,}"  # Matches two or more consecutive newlines
    paragraphs = re.split(pattern, text)
    return paragraphs



def embed_sentence(sentence):
    """Embeds a sentence into a vector using the sentence transformer model"""
    model = SentenceTransformer('all-mpnet-base-v2') 
    sentence_embeddings = model.encode(sentences=[sentence])
    return sentence_embeddings[0]      

# Function to search for relevant passages based on contextual similarity using sentence transformers
def search_with_similarity(query, text):
    # Load pre-trained sentence transformer model
    
    passages = []
    query_embedding = embed_sentence(query)

    for paragraph in text:
        passage_embedding = embed_sentence(paragraph)
        similarity_score = util.cos_sim(query_embedding, passage_embedding)
        passages.append((paragraph, similarity_score))

    passages.sort(key=lambda x: x[1], reverse=True)
    return passages[:2]

# Function to generate text based on retrieved passages using GPT-2 model
def generate_response(passages, max_length=130, temperature=0, top_k=50, top_p=0.95, num_return_sequences=1):
    # Load pre-trained GPT model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    combined_passages = " ".join([passage[0] for passage in passages])
    input_text = combined_passages[:512]
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, 
                                temperature=temperature, top_k=top_k, top_p=top_p, 
                                num_return_sequences=num_return_sequences, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("RAG")

query = st.text_input("Enter your query:")
if st.button("Generate Answer"):
    # Load PDF
    pdf_path = r"C:\Users\chaitanya\Downloads\areete\Knowledge base for RAG-Handbook-of-Good-Dairy-Husbandry-Practices_.pdf" # Replace with the path to your PDF file

    pdf_text = extract_text_from_pdf(pdf_path)
    text = segment_text(pdf_text)
    # Search for relevant passages based on the query
    relevant_passages = search_with_similarity(query, text)

    # Generate response based on retrieved passages
    generated_response = generate_response(relevant_passages)

    # Display generated response
    st.subheader("Generated Response")
    st.write(generated_response)


