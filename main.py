import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Đường dẫn tới dữ liệu
train_dir = 'anh_dong_vat/Train'
validation_dir = 'anh_dong_vat/Validation'


# Hàm để tải và tiền xử lý ảnh
def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = Image.open(img_path)
    img = img.resize(target_size)  # Thay đổi kích thước ảnh
    img_array = np.array(img) / 255.0  # Chuẩn hóa pixel về [0, 1]
    return img_array


# Hàm tạo bộ dữ liệu từ thư mục
def generate_data_from_directory(directory, batch_size=32, target_size=(150, 150)):
    images = []
    labels = []
    class_names = os.listdir(directory)

    # Lặp qua từng lớp trong thư mục
    for class_name in class_names:
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg')):  # Kiểm tra ảnh
                    img_array = load_and_preprocess_image(img_path, target_size)
                    images.append(img_array)
                    labels.append(class_names.index(class_name))  # Gán nhãn lớp

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# Tạo bộ dữ liệu huấn luyện và kiểm tra
train_images, train_labels = generate_data_from_directory(train_dir)
validation_images, validation_labels = generate_data_from_directory(validation_dir)

# Xây dựng mô hình CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Dành cho phân loại nhị phân
])

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    train_images, train_labels,
    batch_size=32,
    epochs=20,
    validation_data=(validation_images, validation_labels)
)

# Vẽ đồ thị kết quả huấn luyện
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(20)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(validation_images, validation_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")


# Dự đoán hình ảnh mới (chó hoặc mèo)
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Không tìm thấy ảnh tại đường dẫn: {img_path}")
        return

    img = Image.open(img_path)
    img = img.resize((150, 150))  # Đọc và thay đổi kích thước ảnh
    img_array = np.array(img) / 255.0  # Chuyển thành mảng numpy và chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

    prediction = model.predict(img_array)  # Dự đoán
    if prediction[0] > 0.5:
        print("Dự đoán: Chó")
    else:
        print("Dự đoán: Mèo")


# Ví dụ dự đoán cho ảnh gửi vào
predict_image(r"D:\Kì 2\Lập trình ứng dụng bằng Python\Bài tập\CNN\anh_dong_vat\Train\Cats\cat.1.jpg")  # Thay bằng đường dẫn ảnh kiểm tra (chó hoặc mèo)
