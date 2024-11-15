import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import cv2

# Đường dẫn dữ liệu
train_dir = 'anh_dong_vat/Train'
validation_dir = 'anh_dong_vat/Validation'

# Tiền xử lý ảnh
def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = Image.open(img_path).convert('RGB')  # Đảm bảo ảnh luôn ở chế độ RGB
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

# Sinh dữ liệu từ thư mục
def generate_data_from_directory(directory, target_size=(150, 150)):
    images = []
    labels = []
    class_names = os.listdir(directory)

    for class_name in class_names:
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Tải ảnh
                    img_array = load_and_preprocess_image(img_path, target_size)
                    images.append(img_array)
                    labels.append(class_names.index(class_name))

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels

# Tạo dữ liệu huấn luyện và kiểm tra
train_images, train_labels = generate_data_from_directory(train_dir)
validation_images, validation_labels = generate_data_from_directory(validation_dir)

# Xây dựng mô hình CNN
cnn_model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax')  # Chỉnh thành 2 lớp đầu ra cho phân loại nhị phân
])

# Biên dịch mô hình
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = cnn_model.fit(
    train_images, train_labels,
    epochs=20, batch_size=32,
    validation_data=(validation_images, validation_labels)
)

# Đánh giá mô hình CNN
print("Đánh giá trên tập kiểm tra (CNN):")
test_loss, test_accuracy = cnn_model.evaluate(validation_images, validation_labels)
print(f"Độ chính xác trên tập kiểm tra (CNN): {test_accuracy * 100:.2f}%")
print(f"Loss trên tập kiểm tra (CNN): {test_loss:.4f}")

# Vẽ đồ thị kết quả huấn luyện
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(20), acc, label='Training Accuracy')
plt.plot(range(20), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy (CNN)')

plt.subplot(1, 2, 2)
plt.plot(range(20), loss, label='Training Loss')
plt.plot(range(20), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss (CNN)')
plt.show()

# Báo cáo chi tiết cho CNN
predictions = np.argmax(cnn_model.predict(validation_images), axis=-1)
print("Báo cáo phân loại (CNN):")
print(classification_report(validation_labels, predictions, target_names=['Cat', 'Dog']))

# Hàm dự đoán cho CNN
def predict_image_cnn(img_path):
    img_array = load_and_preprocess_image(img_path)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = cnn_model.predict(img_array)
    label = "Dog" if np.argmax(prediction) == 1 else "Cat"
    print(f"Ảnh (CNN): {img_path}, Dự đoán (CNN): {label}, Giá trị đầu ra: {prediction[0][np.argmax(prediction)]:.4f}")
    return label

# Test dự đoán CNN
print("\nDự đoán từng ảnh cụ thể (CNN):")
predict_image_cnn(r"C:\Users\Admin\Downloads\anh_dong_vat\Validation\Cats\cat.4648.jpg")
predict_image_cnn(r"C:\Users\Admin\Downloads\anh_dong_vat\Validation\Dogs\dog_18.jpg")

# ------------------ Phần R-CNN ------------------

# Hàm tạo các vùng quan tâm bằng Sliding Window
def sliding_window(img, step_size, window_size):
    """Trả về tọa độ các cửa sổ (x, y, w, h)."""
    regions = []
    for y in range(0, img.shape[0] - window_size[1], step_size):
        for x in range(0, img.shape[1] - window_size[0], step_size):
            regions.append((x, y, window_size[0], window_size[1]))
    return regions

# Tiền xử lý với Sliding Window
def extract_regions_sliding_window(img_path, step_size=50, window_size=(150, 150)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không thể tải ảnh từ đường dẫn: {img_path}")

    regions = sliding_window(img, step_size, window_size)
    region_images = []
    for (x, y, w, h) in regions[:10]:  # Chỉ lấy 10 vùng đầu tiên
        roi = img[y:y + h, x:x + w]
        if roi.size > 0:  # Đảm bảo ROI không rỗng
            roi = cv2.resize(roi, (150, 150))
            region_images.append(roi / 255.0)
    return np.array(region_images)

# Hàm dự đoán cho các vùng quan tâm R-CNN
def predict_rcnn_regions(img_path, cnn_model, step_size=50, window_size=(150, 150)):
    # Tiền xử lý với Sliding Window
    regions = extract_regions_sliding_window(img_path, step_size, window_size)

    # Dự đoán từng vùng quan tâm (ROI)
    predictions = []
    for i, region in enumerate(regions):
        region = np.expand_dims(region, axis=0)  # Thêm batch dimension
        prediction = cnn_model.predict(region)
        label = "Dog" if np.argmax(prediction) == 1 else "Cat"
        print(f"Vùng {i + 1} (R-CNN) - Dự đoán: {label}, Giá trị đầu ra: {prediction[0][np.argmax(prediction)]:.4f}")
        predictions.append(label)

    return predictions

# Test dự đoán R-CNN
try:
    img_path = r"C:\Users\Admin\Downloads\anh_dong_vat\Validation\Cats\cat.4648.jpg"
    print("\nDự đoán cho các vùng quan tâm (R-CNN):")
    predictions = predict_rcnn_regions(img_path, cnn_model)
    print(f"Dự đoán cho các vùng quan tâm (R-CNN): {predictions}")
except Exception as e:
    print(f"Lỗi xảy ra: {e}")
