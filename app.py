import os
import re
import string
from collections import Counter
from flask import Flask, render_template, request, current_app
from werkzeug.utils import secure_filename
from flask_cors import CORS
import spacy
from spacy import displacy
from sklearn.feature_extraction.text import TfidfVectorizer
import en_core_web_sm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Initialize NLTK with all required resources
def initialize_nltk():
    required_resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon',
        'averaged_perceptron_tagger'
    ]
    
    for resource in required_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource)

initialize_nltk()

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MULTI_UPLOAD_FOLDER'] = 'multi_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MULTI_UPLOAD_FOLDER'], exist_ok=True)

# Load NLP model
nlp = en_core_web_sm.load()

def preprocess_text(text):
    """Enhanced text preprocessing with NLTK"""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{string.punctuation}]", " ", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in STOPWORDS]
    return tokens

def parse_chat(filepath):
    """Parse chat file into user and AI messages"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    user_msgs = []
    ai_msgs = []

    for line in lines:
        if line.startswith("User:"):
            user_msgs.append(line[5:].strip())
        elif line.startswith("AI:"):
            ai_msgs.append(line[3:].strip())

    return user_msgs, ai_msgs

def analyze_sentiment(texts):
    """Perform sentiment analysis using NLTK's VADER"""
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for text in texts:
        scores = sia.polarity_scores(text)
        if scores['compound'] >= 0.05:
            sentiments.append(1)  # Positive
        elif scores['compound'] <= -0.05:
            sentiments.append(-1)  # Negative
        else:
            sentiments.append(0)  # Neutral
    
    return {
        'positive': len([s for s in sentiments if s > 0]),
        'neutral': len([s for s in sentiments if s == 0]),
        'negative': len([s for s in sentiments if s < 0])
    }

def extract_entities(texts):
    """Extract named entities using spaCy"""
    entities = []
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
    return Counter(entities).most_common(5)

def extract_keywords_nlp(texts, top_n=10):
    """Enhanced keyword extraction with NLTK"""
    all_text = ' '.join(texts)
    tokens = preprocess_text(all_text)
    
    # Get frequency distribution
    freq_dist = FreqDist(tokens)
    
    # Filter short words and get most common
    keywords = [(word, count) for word, count in freq_dist.most_common() 
               if len(word) > 2][:top_n]
    
    return keywords

def generate_conversation_summary(user_msgs, ai_msgs, keywords, topics):
    """Generate a clear summary of the conversation"""
    summary_lines = []
    
    # Basic statistics
    summary_lines.append(f"- The conversation had {len(user_msgs) + len(ai_msgs)} exchanges")
    
    # Nature of conversation based on topics
    if topics:
        main_topics = ", ".join([word for topic in topics[:2] for word in topic[:3]])
        summary_lines.append(f"- The discussion mainly revolved around: {main_topics}")
    elif keywords:
        main_keywords = ", ".join([word for word, count in keywords[:3]])
        summary_lines.append(f"- The user asked mainly about: {main_keywords}")
    
    # Most common keywords
    if keywords:
        keyword_str = ", ".join([f"{word}" for word, count in keywords[:5]])
        summary_lines.append(f"- Most common keywords: {keyword_str}")
    
    return "\n".join(summary_lines)

def identify_topics(texts, num_topics=3):
    """Identify main topics using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                   tokenizer=preprocess_text)
        tfidf = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        topics = []
        for i in range(min(num_topics, len(texts))):
            top_features = tfidf[i].indices[tfidf[i].data.argsort()[-5:][::-1]]
            topics.append([feature_names[j] for j in top_features])
        
        return topics
    except Exception:
        return []

def summarize_chat_log(filepath):
    """Generate comprehensive NLP analysis of chat log with consistent return structure"""
    try:
        user_msgs, ai_msgs = parse_chat(filepath)
        all_msgs = user_msgs + ai_msgs
        
        if not all_msgs:
            return {"error": "No valid messages found in the chat log", "filename": os.path.basename(filepath)}
        
        # Basic statistics
        stats = {
            "total_messages": len(all_msgs),
            "user_count": len(user_msgs),
            "ai_count": len(ai_msgs),
            "avg_msg_length": sum(len(msg) for msg in all_msgs) / len(all_msgs) if all_msgs else 0,
            "filename": os.path.basename(filepath)
        }
        
        # NLP Analysis
        keywords = extract_keywords_nlp(all_msgs)
        topics = identify_topics(all_msgs)
        
        nlp_results = {
            "sentiment": analyze_sentiment(all_msgs),
            "named_entities": extract_entities(all_msgs),
            "keywords": keywords,
            "topics": topics,
            "conversation_summary": generate_conversation_summary(user_msgs, ai_msgs, keywords, topics),
            "example_user_msg": user_msgs[0] if user_msgs else None,
            "example_ai_msg": ai_msgs[0] if ai_msgs else None
        }
        
        return {**stats, **nlp_results}
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}", "filename": os.path.basename(filepath)}

def process_multiple_logs(folder_path):
    """Process all chat logs in a folder with proper error handling"""
    summaries = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            try:
                summary = summarize_chat_log(filepath)
                # Only include valid summaries that have the expected structure
                if isinstance(summary, dict) and 'total_messages' in summary:
                    summaries.append(summary)
                else:
                    print(f"Skipping invalid summary for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    if not summaries:
        return {"error": "No valid chat logs found in folder"}
    
    try:
        combined = {
            "total_files": len(summaries),
            "total_messages": sum(s.get('total_messages', 0) for s in summaries),
            "avg_messages_per_file": sum(s.get('total_messages', 0) for s in summaries) / len(summaries),
            "most_common_keywords": get_combined_keywords(summaries),
            "file_summaries": summaries
        }
        return combined
    except Exception as e:
        return {"error": f"Error generating combined summary: {str(e)}"}

def get_combined_keywords(summaries):
    """Combine keywords from multiple chat logs"""
    all_keywords = []
    for summary in summaries:
        if 'keywords' in summary:
            all_keywords.extend([kw[0] for kw in summary['keywords']])
    
    return Counter(all_keywords).most_common(10)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    multi_summary = None
    
    if request.method == 'POST':
        # Check for single file upload
        if 'chatfile' in request.files:
            file = request.files['chatfile']
            if file and file.filename.endswith('.txt'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                summary = summarize_chat_log(filepath)
        
        # Check for folder upload
        elif 'chatfolder' in request.files:
            files = request.files.getlist('chatfolder')
            for file in files:
                if file and file.filename.endswith('.txt'):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['MULTI_UPLOAD_FOLDER'], filename)
                    file.save(filepath)
            multi_summary = process_multiple_logs(app.config['MULTI_UPLOAD_FOLDER'])
    
    return render_template('index.html', summary=summary, multi_summary=multi_summary)

if __name__ == '__main__':
    # Run the app with host and port configuration
    app.run(host='0.0.0.0', port=5000, debug=True)