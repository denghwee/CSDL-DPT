import os
import joblib
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2

# Load model VGG16
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load Haar Cascade Classifier để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dữ liệu đã lưu
all_images = joblib.load('all_imag.jbl')
attribute_list = np.array(joblib.load('attribute.jbl'))


def extract_features(img_path):
    """Trích xuất đặc trưng từ ảnh bằng VGG-16, với phát hiện khuôn mặt hoặc xử lý trực tiếp nếu là ảnh đã cắt"""
    try:
        if not os.path.exists(img_path):
            print(f"File does not exist: {img_path}")
            return None

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {img_path}: Image is None")
            return None

        # Kiểm tra xem ảnh có nằm trong cropped_faces (dựa trên tên file)
        filename = os.path.basename(img_path)
        is_cropped_face = any(filename in img_path_full for img_path_full in all_images)

        if is_cropped_face:
            print(f"Processing pre-cropped image: {img_path}")
            img_resized = cv2.resize(img, (224, 224))
        else:
            print(f"Detecting face in new image: {img_path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                print(f"No faces detected in {img_path}")
                return None

            x, y, w, h = faces[0]
            face_img = img[y:y + h, x:x + w]
            if face_img.size == 0:
                print(f"Invalid face region in {img_path}")
                return None
            img_resized = cv2.resize(face_img, (224, 224))

        array_of_image = np.array(img_resized, dtype=np.float32)
        expand_img = np.expand_dims(array_of_image, axis=0)
        preproces_img = preprocess_input(expand_img)
        result = model.predict(preproces_img).flatten()
        print(f"Features extracted for {img_path}")
        return result

    except Exception as e:
        print(f"General error processing image {img_path}: {str(e)}")
        return None


def find_similar_images(query_features):
    """Tìm kiếm 3 ảnh tương tự nhất và trả về đường dẫn cùng tên nhân vật"""
    if query_features is None or len(query_features) == 0:
        print("Query features are invalid")
        return []

    similarities = []
    for i in range(len(attribute_list)):
        sim_score = cosine_similarity(query_features.reshape(1, -1), attribute_list[i].reshape(1, -1))[0][0]
        similarities.append((i, sim_score))

    top_indices = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

    results = []
    for idx, score in top_indices:
        img_path = all_images[idx]
        img_path = img_path.replace('\\', '/')
        character_name = os.path.basename(os.path.dirname(img_path))
        display_path = img_path.replace('static/', '')
        results.append({
            'path': display_path,
            'character': character_name,
            'score': float(score)
        })

    return results


def find_similar_images_sql(query_features, top_k=3):
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS')
    )
    with conn.cursor() as cur:
        # Chuyển vector numpy sang chuỗi cho truy vấn pgvector
        vector_str = '[' + ','.join(str(float(x)) for x in query_features) + ']'
        cur.execute('''
            SELECT i.image_path, i.name, f.feature_vector <=> %s::vector AS distance
            FROM image_features f
            JOIN images i ON f.image_id = i.id
            WHERE f.model_name = 'VGG16'
            ORDER BY distance ASC
            LIMIT %s;
        ''', (vector_str, top_k))
        results = cur.fetchall()
    conn.close()
    # Trả về danh sách dict giống hệt find_similar_images
    formatted_results = []
    for row in results:
        img_path = row[0].replace('\\', '/').replace('static/', '')
        character_name = row[1]
        display_path = img_path  # đã loại static/ ở trên
        formatted_results.append({
            'path': display_path,
            'character': character_name,
            'score': float(1-row[2])
        })
    return formatted_results