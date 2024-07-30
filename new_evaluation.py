import json
import bert_score
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()

# Load the pre-trained model
model = bert_score.BERTScorer(model_type='bert-base-uncased')

# Load the intents JSON file
intents_file = open('final-intents.json').read()
intents = json.loads(intents_file)

def clean_up_sentence(sentence):
    # Tokenize the pattern - Split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word - Create base word, in an attempt to represent related words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def evaluate_intent(intent):
    patterns = intent['patterns']
    responses = intent['responses']

    generated_responses = []
    target_responses = []

    for pattern in patterns:
        cleaned_pattern = ' '.join(clean_up_sentence(pattern))
        target_responses.extend(responses)

        # Generate a response using the chatbot
        generated_response = chatbot_response(cleaned_pattern)
        generated_responses.extend([generated_response] * len(responses))

    # Calculate BERTScore
    _, _, F1 = bert_score.score(generated_responses, target_responses, model, verbose=False)

    # Print the evaluation results
    print(f"Intent: {intent['tag']}")
    print(f"BERTScore F1-score : {F1.mean().item():.4f}")
    print()

def evaluate_chatbot():
    for intent in intents['intents']:
        evaluate_intent(intent)

def clean_up_sentence(sentence):
    # Tokenize the pattern - Split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word - Create base word, in an attempt to represent related words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def chatbot_response(msg):
    # Implement your chatbot response logic here
    # Replace this placeholder with your actual chatbot implementation
    return "Placeholder response"

# Run the evaluation
evaluate_chatbot()
