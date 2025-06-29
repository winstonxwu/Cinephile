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

def load_conversations():
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
    
def extract_qa_pairs(lines, conversations):
    qa_pairs = []
    for conv_ids in conversations:
        for i in range(len(conv_ids)-1):
            input_line = lines.get(conv_ids[i])
            response_line = lines.get(conv_ids[i+1])
            if input_line and response_line:
                qa_pairs.append((input_line, response_line))
    return qa_pairs

print("Loading lines ...")
lines_dict = load_lines()
print(f"Loaded {len(lines_dict)} lines.")

print("Loading conversations...")
conversations_list = load_conversations()
print(f"Loaded {len(conversations_list)} conversations.")

print("Extracting QA pairs...")
qa_data = extract_qa_pairs(lines_dict, conversations_list)
print(f"Extracted {len(qa_data)} QA pairs.")

if qa_data:
    print("\nExample QA pair:")
    print(f"Input: {qa_data[0][0]}")
    print(f"Response: {qa_data[0][1]}")