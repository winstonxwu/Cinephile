import os
import re

DATA_PATH = '/Users/winstonxwu/Downloads/archive'
LINES_FILE = os.path.join(DATA_PATH, 'movie_lines.txt')
CONVERSATIONS_FILE = os.path.join(DATA_PATH, 'movie_conversations.txt')

def load_lines():
    lines = {}
    with open(LINES_FILE, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                line_id = parts[0]
                text = parts[4].strip()
                lines[line_id] = text
    return lines

def load_conversation():
    conversations = []
    with open(CONVERSATIONS_FILE, 'r', encoding='iso-8859-1') as f:
        conversations = []
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                line_ids_str = parts[3].strip()
                try:
                    line_ids = eval(line_ids_str)
                    conversations.append(line_ids)
                except SyntaxError:
                    print(f"Skipping malformed line ID String: {line_ids_str}")
        return conversations
    
# def extract_qa_pairs(lines, conversations):
    