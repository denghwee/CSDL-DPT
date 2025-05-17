import os
import psycopg2
from dotenv import load_dotenv
import joblib as jb
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
import torch

# Load environment variables from .env file
load_dotenv()
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')

# Path to the cropped faces directory
CROPPED_FACES_DIR = os.path.join('static', 'cropped_faces')

def connect_db():
    return psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL UNIQUE,
                image_type TEXT,
                uploaded_at TIMESTAMP DEFAULT NOW(),
                description TEXT,
                description_vector VECTOR(512)
            );
        ''')
        conn.commit()

def insert_image(conn, name, image_path, image_type, description, description_vector=None):
    with conn.cursor() as cur:
        cur.execute('''
            INSERT INTO images (name, image_path, image_type, description, description_vector)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        ''', (name, image_path, image_type, description, description_vector))
        image_id = cur.fetchone()[0]
        conn.commit()
        return image_id

def create_features_table(conn):
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS image_features (
                id SERIAL PRIMARY KEY,
                image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                model_name TEXT NOT NULL,
                feature_vector VECTOR(512),
                extracted_at TIMESTAMP DEFAULT NOW()
            );
        ''')
        conn.commit()

def insert_feature(conn, image_id, model_name, feature_vector):
    with conn.cursor() as cur:
        cur.execute('''
            INSERT INTO image_features (image_id, model_name, feature_vector)
            VALUES (%s, %s, %s)
            RETURNING id;
        ''', (image_id, model_name, feature_vector))
        conn.commit()

def get_clip_text_embedding(text, tokenizer, text_model):
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def upload_all_images():
    conn = connect_db()
    create_table(conn)
    create_features_table(conn)
    # Load CLIP model for 512-dim text embedding
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    # Load all feature vectors from attribute.jbl
    attribute_list = np.array(jb.load(open("attribute.jbl",'rb')))
    image_idx = 0
    for name in os.listdir(CROPPED_FACES_DIR):
        char_dir = os.path.join(CROPPED_FACES_DIR, name)
        if not os.path.isdir(char_dir):
            continue
        for filename in os.listdir(char_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(char_dir, filename)
                image_type = filename.split('.')[-1]
                description = f"Face of {name}"

                # Generate CLIP embedding for description (512-dim)
                vec = get_clip_text_embedding(description, tokenizer, text_model)
                vec = [float(x) for x in vec]

                # Insert image into database
                print(f'Uploading {image_path}...')
                image_id = insert_image(conn, name, image_path, image_type, description, vec)

                # Lưu feature_vector từ attribute_list vào bảng features
                feature_vector = [float(x) for x in attribute_list[image_idx]]
                insert_feature(conn, image_id, 'VGG16', feature_vector)
                image_idx += 1
    conn.close()
    print('All images uploaded.')

if __name__ == '__main__':
    upload_all_images()
