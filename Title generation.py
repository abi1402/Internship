!pip install transformers
from transformers import BartTokenizer, BartForConditionalGeneration

def generate_title_from_paragraph(paragraph):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + paragraph, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    title = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return title

if __name__ == "__main__":
    paragraph = """
    Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction
    between humans and computers using natural language. It aims to enable computers to understand, interpret, and 
    generate human language in a way that is both meaningful and contextually relevant. NLP plays a vital role in 
    various applications, such as machine translation, sentiment analysis, text summarization, and chatbots.

    In recent years, NLP has witnessed significant advancements, especially with the advent of deep learning 
    techniques and large-scale language models like GPT-3 and BERT. These models have demonstrated exceptional 
    language understanding and generation capabilities, leading to groundbreaking improvements in NLP tasks.

    This code snippet demonstrates how to utilize the BART model, a powerful transformer-based architecture,
    for title generation from a given paragraph. By leveraging the model's summarization capabilities, it can
    extract key information from the paragraph and generate a concise and meaningful title.

    Give it a try and see how well the BART model can summarize the provided paragraph and generate an appropriate title!
    """

    reference_title = "The Power of Natural Language Processing with BART Model"
    generated_title = generate_title_from_paragraph(paragraph)
    print("Generated Title:", generated_title)
    
    # Calculate accuracy
    accuracy = 1.0 if generated_title.lower() == reference_title.lower() else 0.0
    print("Accuracy:", accuracy)

