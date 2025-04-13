# app.py
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# Download necessary NLTK data
nltk.download('punkt')

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

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.get_json()
    text = data.get('text', '')
    method = data.get('method', 'nltk_word')
    
    tokens = []
    if not text:
        return jsonify({'tokens': [], 'count': 0, 'error': 'No text provided'})
    
    try:
        if method == 'nltk_word':
            tokens = word_tokenize(text)
        elif method == 'nltk_sentence':
            tokens = sent_tokenize(text)
        elif method == 'spacy':
            if spacy_available:
                doc = nlp(text)
                tokens = [token.text for token in doc]
            else:
                return jsonify({'tokens': [], 'count': 0, 'error': 'SpaCy is not available. Please install it for this feature.'})
        elif method == 'regex_word':
            tokens = re.findall(r'\b\w+\b', text)
        elif method == 'regex_sentence':
            tokens = re.split(r'(?<=[.!?])\s+', text)
        # Speech tokenization simply uses NLTK word tokenization on the speech-to-text result
        elif method == 'speech':
            tokens = word_tokenize(text)
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
            # Basic analysis without spaCy
            words = word_tokenize(text.lower())
            # Filter out punctuation
            words = [word for word in words if re.match(r'\w+', word)]
            
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