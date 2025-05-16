from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
from model import extract_features, find_similar_images

app = Flask(__name__)

# Cấu hình thư mục upload
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Cấu hình session
app.secret_key = 'your_secret_key_here'  # Cần thiết để sử dụng flash messages và session

# Kiểm tra định dạng file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    # Khởi tạo biến để truyền vào template
    results = session.get('results', [])  # Lấy kết quả từ session (nếu có)
    uploaded_image = session.get('uploaded_image', None)  # Lấy ảnh upload từ session
    error_message = None

    if request.method == 'POST':
        print("Received POST request")  # Log

        # Xử lý upload ảnh
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                uploaded_image = filename
                print(f"Uploaded image saved at: {file_path}")  # Log

                # Trích xuất đặc trưng
                features = extract_features(file_path)
                if features is not None:
                    print("Features extracted successfully")  # Log
                    similar_images = find_similar_images(features)
                    results = similar_images[:3]  # Lấy 3 ảnh tương tự nhất
                    print(f"Found similar images: {results}")  # Log
                else:
                    error_message = "Không thể xử lý ảnh đầu vào. Vui lòng đảm bảo ảnh là khuôn mặt hợp lệ."
                    flash(error_message, 'error')  # Sử dụng flash để hiển thị thông báo lỗi
                    print("Failed to extract features")  # Log

                # Lưu kết quả vào session (đã được chuyển đổi sang float)
                session['results'] = results
                session['uploaded_image'] = uploaded_image

        # Xử lý reset
        if 'reset' in request.form:
            print("Reset requested")  # Log
            # Xóa ảnh đã upload
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
            # Xóa session
            session.pop('results', None)
            session.pop('uploaded_image', None)
            flash("Đã xóa kết quả và ảnh upload.", 'info')
            return redirect(url_for('index'))

        return redirect(url_for('index'))  # Redirect để tránh refresh không mong muốn

    print("Rendering template")  # Log
    return render_template('index.html', results=results, uploaded_image=uploaded_image)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)