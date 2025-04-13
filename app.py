# app.py
from flask import Flask, render_template, request, jsonify
import re

# Define regex patterns for tokenization
word_pattern = re.compile(r'\b\w+\b')
sentence_pattern = re.compile(r'(?<=[.!?])\s+')

# Try to import spaCy, but make it optional
try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
        spacy_available = True
    except:
        print("SpaCy model not found. Please install it using: python -m spacy download en_core_web_sm")
        spacy_available = False
except ImportError:
    print("SpaCy not installed. Advanced NLP features will be disabled.")
    spacy_available = False
    nlp = None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', spacy_available=spacy_available)

def regex_word_tokenize(text):
    """Custom word tokenizer using regex"""
    return word_pattern.findall(text)

def regex_sentence_tokenize(text):
    """Custom sentence tokenizer using regex"""
    # Add the first sentence too by splitting and handling edge cases
    if not text:
        return []
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.get_json()
    text = data.get('text', '')
    method = data.get('method', 'regex_word')
    
    tokens = []
    if not text:
        return jsonify({'tokens': [], 'count': 0, 'error': 'No text provided'})
    
    try:
        if method == 'nltk_word' or method == 'regex_word' or method == 'speech':
            # Use regex for all word tokenization methods
            tokens = regex_word_tokenize(text)
        elif method == 'nltk_sentence' or method == 'regex_sentence':
            # Use regex for all sentence tokenization methods
            tokens = regex_sentence_tokenize(text)
        elif method == 'spacy':
            if spacy_available:
                doc = nlp(text)
                tokens = [token.text for token in doc]
            else:
                return jsonify({'tokens': [], 'count': 0, 'error': 'SpaCy is not available. Please install it for this feature.'})
        else:
            return jsonify({'tokens': [], 'count': 0, 'error': 'Invalid tokenization method'})
        
        return jsonify({
            'tokens': tokens,
            'count': len(tokens),
            'error': None
        })
    except Exception as e:
        return jsonify({'tokens': [], 'count': 0, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'})
    
    try:
        # Only provide NLP analysis if spaCy is available
        if spacy_available:
            doc = nlp(text)
            
            # Extract entities
            entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
            
            # Get part-of-speech tags
            pos_tags = [{'text': token.text, 'pos': token.pos_} for token in doc]
            
            # Create word frequency count
            word_freq = {}
            for token in doc:
                if not token.is_punct and not token.is_space:
                    word = token.text.lower()
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
            
            return jsonify({
                'entities': entities,
                'pos_tags': pos_tags,
                'word_frequency': word_freq,
                'error': None
            })
        else:
            # Basic analysis without spaCy or NLTK
            words = regex_word_tokenize(text.lower())
            
            # Create word frequency count
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
            
            return jsonify({
                'entities': [],
                'pos_tags': [],
                'word_frequency': word_freq,
                'error': 'SpaCy is not available. Only basic analysis is provided.'
            })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)