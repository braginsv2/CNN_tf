import os
import glob
import numpy as np
import tensorflow as tf
import cv2

# Константы (должны совпадать с обучением)
CLASSES = 8
SAMPLE_SIZE = (256, 256)

# Функции для создания модели (копируем из основного кода)
def input_layer():
    return tf.keras.layers.Input(shape=SAMPLE_SIZE + (3,))

def downsample_block(filters, size, batch_norm=True):
    initializer = tf.keras.initializers.GlorotNormal()
    result = tf.keras.Sequential()
    
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample_block(filters, size, dropout=False):
    initializer = tf.keras.initializers.GlorotNormal()
    result = tf.keras.Sequential()
    
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                        kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())
    
    if dropout:
        result.add(tf.keras.layers.Dropout(0.25))
    
    result.add(tf.keras.layers.ReLU())
    return result

def output_layer(size):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv2DTranspose(CLASSES, size, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='sigmoid')

# Создаем модель
print("Создаем архитектуру модели...")
inp_layer = input_layer()

downsample_stack = [
    downsample_block(64, 4, batch_norm=False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

upsample_stack = [
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]

out_layer = output_layer(4)

# Реализуем skip connections
x = inp_layer
downsample_skips = []

for block in downsample_stack:
    x = block(x)
    downsample_skips.append(x)
    
downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
    x = up_block(x)
    x = tf.keras.layers.Concatenate()([x, down_block])

out_layer = out_layer(x)

unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)

# Загружаем обученные веса
print("Загружаем обученные веса...")
try:
    unet_like.load_weights('unet_like')
    print("Веса загружены успешно!")
except Exception as e:
    print(f"Ошибка загрузки весов: {e}")
    print("Убедитесь, что файл 'unet_like' существует в текущей папке")
    exit(1)

# Цвета для каждого класса (BGR для OpenCV)
bgr_colors = [
    (0,   0,   0),      # Класс 0 - черный (фон)
    (0,   0,   255),    # Класс 1 - красный
    (0,   255, 0),      # Класс 2 - зеленый
    (255, 0,   0),      # Класс 3 - синий
    (0,   165, 255),    # Класс 4 - оранжевый
    (203, 192, 255),    # Класс 5 - розовый
    (255, 255, 0),      # Класс 6 - циан
    (255, 0,   255)     # Класс 7 - пурпурный
]

def resize_image(image, target_size):
    """Изменение размера изображения"""
    return cv2.resize(image, target_size)

def find_contours_simple(mask, threshold=0.5):
    """Простой поиск контуров"""
    # Преобразуем в бинарную маску
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Находим контуры
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def process_video_frames():
    """Основная функция обработки кадров"""
    
    # Создаем папку для результатов, если её нет
    os.makedirs('videos/processed', exist_ok=True)
    
    # Получаем список кадров для обработки
    frames = sorted(glob.glob('videos/original_video/*.jpg'))
    print(f"Найдено {len(frames)} кадров для обработки")
    
    if len(frames) == 0:
        print("Ошибка: не найдены кадры в папке 'videos/original_video/'")
        print("Убедитесь, что кадры сохранены в правильной папке")
        return
    
    print("Начинаем обработку кадров...")
    
    for i, filename in enumerate(frames):
        try:
            # Загружаем кадр
            frame = cv2.imread(filename)
            original_height, original_width = frame.shape[:2]
            
            # Изменяем размер для предсказания (OpenCV использует (width, height))
            sample = cv2.resize(frame, (SAMPLE_SIZE[1], SAMPLE_SIZE[0]))
            sample = sample.astype(np.float32) / 255.0
            
            # Получаем предсказание модели
            predict = unet_like.predict(sample.reshape((1,) + SAMPLE_SIZE + (3,)), verbose=0)
            predict = predict.reshape(SAMPLE_SIZE + (CLASSES,))
            
            # Затемняем исходный кадр для лучшей видимости сегментации
            frame = (frame * 0.7).astype(np.uint8)
            
            # Обрабатываем каждый класс (пропускаем фон - класс 0)
            for channel in range(1, CLASSES):
                try:
                    # Изменяем размер маски до исходного размера кадра
                    mask = predict[:,:,channel]
                    mask_resized = cv2.resize(mask, (original_width, original_height))
                    
                    # Находим контуры
                    contours = find_contours_simple(mask_resized, threshold=0.3)
                    
                    # Рисуем контуры
                    cv2.drawContours(frame, contours, -1, bgr_colors[channel], thickness=2)
                    
                    # Дополнительно: заливаем области с прозрачностью
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask_binary == 1] = bgr_colors[channel]
                    
                    # Смешиваем с исходным кадром
                    alpha = 0.3  # Прозрачность заливки
                    frame = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
                    
                except Exception as e:
                    print(f"Ошибка при обработке класса {channel} в кадре {filename}: {e}")
                    continue
            
            # Сохраняем обработанный кадр
            output_filename = f'videos/processed/{os.path.basename(filename)}'
            cv2.imwrite(output_filename, frame)
            
            # Показываем прогресс
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(frames)} кадров")
                
        except Exception as e:
            print(f"Ошибка при обработке кадра {filename}: {e}")
            continue
    
    print(f"Обработка завершена! Результаты сохранены в папке 'videos/processed/'")

def create_video_from_frames(input_folder, output_path, fps=30):
    """Создает видео из обработанных кадров"""
    try:
        # Получаем список обработанных кадров
        processed_frames = sorted(glob.glob(f'{input_folder}/*.jpg'))
        
        if len(processed_frames) == 0:
            print("Нет обработанных кадров для создания видео")
            return
        
        # Читаем первый кадр для определения размеров
        first_frame = cv2.imread(processed_frames[0])
        height, width, layers = first_frame.shape
        
        # Создаем объект VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Создаем видео из {len(processed_frames)} кадров...")
        
        for frame_path in processed_frames:
            frame = cv2.imread(frame_path)
            video_writer.write(frame)
        
        video_writer.release()
        print(f"Видео сохранено: {output_path}")
        
    except Exception as e:
        print(f"Ошибка при создании видео: {e}")

def extract_frames_from_video(video_path, output_folder, max_frames=None):
    """Дополнительная функция: извлечение кадров из видео"""
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and frame_count >= max_frames:
                break
            
            frame_filename = f"{output_folder}/frame_{frame_count:06d}.jpg"
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Извлечено {frame_count} кадров")
        
        cap.release()
        print(f"Извлечено {frame_count} кадров в папку {output_folder}")
        
    except Exception as e:
        print(f"Ошибка при извлечении кадров: {e}")

# Запуск обработки
if __name__ == "__main__":
    # Обрабатываем кадры
    process_video_frames()
    
    # Создаем видео из обработанных кадров
    print("\nСоздание видео из обработанных кадров...")
    create_video_from_frames('videos/processed', 'videos/segmented_video.mp4', fps=30)
    
    print("\nПроцесс завершен!")
    print("Результаты:")
    print("- Обработанные кадры: videos/processed/")
    print("- Итоговое видео: videos/segmented_video.mp4")
    
    # Дополнительно: если у вас есть видеофайл, можно извлечь из него кадры
    # extract_frames_from_video('input_video.mp4', 'videos/original_video', max_frames=1000)