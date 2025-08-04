import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
from collections import Counter

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

# Фиксируем семена
np.random.seed(42)
tf.random.set_seed(42)

def parse_coco_annotations(annotations_file, images_dir):
    """
    Парсит COCO аннотации и извлекает bounding boxes.
    Также находит изображения без аннотаций (считает их как "без кружек")
    """
    print("📋 ПАРСИНГ COCO АННОТАЦИЙ И ПОИСК ВСЕХ ИЗОБРАЖЕНИЙ")
    print("="*50)
    
    # Загружаем COCO данные
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Создаем словари для быстрого поиска
    images_dict = {img['id']: img for img in coco_data['images']}
    coco_filenames = {img['file_name'] for img in coco_data['images']}
    
    # Группируем аннотации по изображениям
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Находим ВСЕ изображения в папке
    all_image_files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        files = glob(os.path.join(images_dir, ext))
        files.extend(glob(os.path.join(images_dir, ext.upper())))
        for f in files:
            all_image_files.add(os.path.basename(f))
    
    print(f"Всего файлов в папке: {len(all_image_files)}")
    print(f"Файлов в COCO аннотациях: {len(coco_filenames)}")
    
    # Подготавливаем данные
    image_data = []
    
    # Сначала обрабатываем изображения С аннотациями
    for image_id, img_info in images_dict.items():
        file_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(file_path):
            continue
            
        # Получаем размеры изображения
        img_width = img_info['width']
        img_height = img_info['height']
        
        if image_id in annotations_by_image:
            # Есть кружки - берем первую аннотацию
            ann = annotations_by_image[image_id][0]
            bbox = ann['bbox']  # [x, y, width, height]
            
            # COCO формат: [x_top_left, y_top_left, width, height]
            # Нормализуем координаты
            x = bbox[0] / img_width
            y = bbox[1] / img_height
            w = bbox[2] / img_width
            h = bbox[3] / img_height
            
            # Конвертируем в центральные координаты
            center_x = x + w / 2
            center_y = y + h / 2
            
            image_data.append({
                'file_name': img_info['file_name'],
                'file_path': file_path,
                'has_cup': 1,
                'bbox': [center_x, center_y, w, h],
                'original_size': [img_width, img_height]
            })
    
    # Теперь обрабатываем изображения БЕЗ аннотаций
    files_without_annotations = all_image_files - coco_filenames
    print(f"Файлов без аннотаций (считаем как 'без кружек'): {len(files_without_annotations)}")
    
    for filename in files_without_annotations:
        file_path = os.path.join(images_dir, filename)
        
        if os.path.exists(file_path):
            try:
                # Получаем размеры изображения
                img = cv2.imread(file_path)
                if img is not None:
                    img_height, img_width = img.shape[:2]
                    
                    image_data.append({
                        'file_name': filename,
                        'file_path': file_path,
                        'has_cup': 0,
                        'bbox': [0.0, 0.0, 0.0, 0.0],  # Нулевой bbox для "нет объекта"
                        'original_size': [img_width, img_height]
                    })
            except Exception as e:
                print(f"⚠️ Ошибка чтения {filename}: {e}")
                continue
    
    # Статистика
    with_cups = sum(1 for item in image_data if item['has_cup'] == 1)
    without_cups = len(image_data) - with_cups
    
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"Всего изображений для обучения: {len(image_data)}")
    print(f"С кружками (из COCO): {with_cups}")
    print(f"Без кружек (без аннотаций): {without_cups}")
    
    if without_cups == 0:
        print("⚠️ ВНИМАНИЕ: Не найдено изображений без кружек!")
        print("   Убедитесь, что в папке есть изображения без аннотаций")
    
    return image_data

def enhanced_preprocessing_detection(image_path, target_size=(224, 224)):
    """
    Предобработка изображения для детекции
    """
    try:
        # Читаем изображение
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Конвертируем в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Изменяем размер с сохранением пропорций
        h, w = image.shape[:2]
        
        # Вычисляем коэффициент масштабирования
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Изменяем размер
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Создаем финальное изображение с padding
        final_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # Центрируем изображение
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2
        final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Нормализация для ImageNet
        normalized = final_image.astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Возвращаем также коэффициенты для денормализации bbox
        return normalized, scale, x_offset, y_offset
        
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
        return None

def adjust_bbox_for_preprocessing(bbox, original_size, scale, x_offset, y_offset, target_size):
    """
    Корректирует координаты bbox после предобработки изображения
    """
    if bbox == [0.0, 0.0, 0.0, 0.0]:
        return bbox
    
    orig_w, orig_h = original_size
    
    # Денормализуем bbox к оригинальным координатам
    center_x = bbox[0] * orig_w
    center_y = bbox[1] * orig_h
    width = bbox[2] * orig_w
    height = bbox[3] * orig_h
    
    # Применяем масштабирование
    center_x = center_x * scale + x_offset
    center_y = center_y * scale + y_offset
    width = width * scale
    height = height * scale
    
    # Нормализуем к размеру target_size
    center_x = center_x / target_size[1]
    center_y = center_y / target_size[0]
    width = width / target_size[1]
    height = height / target_size[0]
    
    return [center_x, center_y, width, height]

def create_detection_dataset(images_dir="train", annotations_file="train/_annotations.coco.json", 
                           target_size=(224, 224), max_samples=None):
    """
    Создает датасет для детекции объектов
    """
    print("🎯 СОЗДАНИЕ ДАТАСЕТА ДЛЯ ДЕТЕКЦИИ")
    print("="*50)
    
    # Парсим аннотации
    image_data = parse_coco_annotations(annotations_file, images_dir)
    
    if max_samples:
        # Сбалансированный выбор образцов
        with_cups = [item for item in image_data if item['has_cup'] == 1]
        without_cups = [item for item in image_data if item['has_cup'] == 0]
        
        print(f"Доступно изображений с кружками: {len(with_cups)}")
        print(f"Доступно изображений без кружек: {len(without_cups)}")
        
        if len(without_cups) == 0:
            print("⚠️ ВНИМАНИЕ: Нет изображений без кружек!")
            print("   Используем только изображения с кружками")
            samples_per_class = min(max_samples, len(with_cups))
            image_data = with_cups[:samples_per_class]
        else:
            # Берем поровну из каждого класса
            samples_per_class = min(max_samples // 2, len(with_cups), len(without_cups))
            selected_with_cups = with_cups[:samples_per_class]
            selected_without_cups = without_cups[:samples_per_class]
            image_data = selected_with_cups + selected_without_cups
            
            print(f"Выбрано с кружками: {len(selected_with_cups)}")
            print(f"Выбрано без кружек: {len(selected_without_cups)}")
    
    print(f"Итого используем: {len(image_data)} изображений")
    
    images = []
    labels = []  # Классификация (есть кружка или нет)
    bboxes = []  # Координаты bounding box
    
    print("📥 Загрузка и обработка изображений...")
    
    for i, item in enumerate(image_data):
        if i % 50 == 0:
            print(f"  Обработано: {i}/{len(image_data)}")
        
        result = enhanced_preprocessing_detection(item['file_path'], target_size)
        if result is not None:
            processed_image, scale, x_offset, y_offset = result
            
            # Корректируем bbox координаты
            adjusted_bbox = adjust_bbox_for_preprocessing(
                item['bbox'], item['original_size'], 
                scale, x_offset, y_offset, target_size
            )
            
            images.append(processed_image)
            labels.append(item['has_cup'])
            bboxes.append(adjusted_bbox)
    
    print(f"\n✅ Успешно загружено: {len(images)} изображений")
    
    # Конвертируем в numpy массивы
    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="float32")
    bboxes = np.array(bboxes, dtype="float32")
    
    print(f"Распределение классов: {Counter(labels)}")
    print(f"Форма данных - изображения: {images.shape}")
    print(f"Форма данных - метки: {labels.shape}")
    print(f"Форма данных - bbox: {bboxes.shape}")
    
    return images, labels, bboxes

def create_detection_model(input_shape=(224, 224, 3)):
    """
    Создает модель для детекции объектов с двумя выходами:
    1. Классификация (есть объект или нет)
    2. Регрессия bbox координат
    """
    print("\n🧠 СОЗДАНИЕ МОДЕЛИ ДЛЯ ДЕТЕКЦИИ")
    print("="*50)
    
    # Базовая модель
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Входной слой
    inputs = Input(shape=input_shape)
    
    # Backbone
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Общие признаки
    features = Dense(512, activation='relu', name='features')(x)
    features = BatchNormalization()(features)
    features = Dropout(0.3)(features)
    
    # Ветка классификации
    classification = Dense(256, activation='relu', name='cls_dense')(features)
    classification = BatchNormalization()(classification)
    classification = Dropout(0.2)(classification)
    classification_output = Dense(1, activation='sigmoid', name='classification')(classification)
    
    # Ветка регрессии bbox
    regression = Dense(256, activation='relu', name='reg_dense')(features)
    regression = BatchNormalization()(regression)
    regression = Dropout(0.2)(regression)
    bbox_output = Dense(4, activation='sigmoid', name='bbox_regression')(regression)  # [center_x, center_y, width, height]
    
    # Создаем модель
    model = Model(inputs=inputs, outputs=[classification_output, bbox_output])
    
    print(f"Общее количество параметров: {model.count_params():,}")
    
    return model

def detection_loss(y_true_cls, y_pred_cls, y_true_bbox, y_pred_bbox):
    """
    Комбинированная функция потерь для детекции
    """
    # Классификационная потеря
    cls_loss = tf.keras.losses.binary_crossentropy(y_true_cls, y_pred_cls)
    
    # Регрессионная потеря (только для изображений с объектами)
    mask = tf.cast(y_true_cls > 0.5, tf.float32)  # Маска для изображений с объектами
    
    # Smooth L1 loss для bbox
    diff = tf.abs(y_true_bbox - y_pred_bbox)
    smooth_l1 = tf.where(
        diff < 1.0,
        0.5 * tf.square(diff),
        diff - 0.5
    )
    
    bbox_loss = tf.reduce_mean(smooth_l1, axis=1)
    bbox_loss = bbox_loss * mask  # Применяем маску
    
    # Объединяем потери
    total_loss = cls_loss + bbox_loss
    
    return total_loss

def custom_detection_loss():
    """
    Кастомная функция потерь для модели с двумя выходами
    """
    def loss_function(y_true, y_pred):
        y_true_cls = y_true[0]
        y_true_bbox = y_true[1]
        y_pred_cls = y_pred[0]
        y_pred_bbox = y_pred[1]
        
        return detection_loss(y_true_cls, y_pred_cls, y_true_bbox, y_pred_bbox)
    
    return loss_function

def train_detection_model():
    """
    Обучение модели детекции
    """
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ ДЕТЕКЦИИ")
    print("="*70)
    
    try:
        # Создаем датасет
        images, labels, bboxes = create_detection_dataset(
            images_dir="train",
            annotations_file="train/_annotations.coco.json",
            target_size=(224, 224),
            max_samples=300  # Ограничиваем для быстрого обучения
        )
        
        if len(images) < 10:
            raise ValueError("Недостаточно данных для обучения!")
        
        # Разделяем данные
        if len(np.unique(labels)) > 1:
            # Есть оба класса - можем использовать стратификацию
            X_train, X_test, y_cls_train, y_cls_test, y_bbox_train, y_bbox_test = train_test_split(
                images, labels, bboxes, test_size=0.2, random_state=42, stratify=labels
            )
            
            X_train, X_val, y_cls_train, y_cls_val, y_bbox_train, y_bbox_val = train_test_split(
                X_train, y_cls_train, y_bbox_train, test_size=0.2, random_state=42, stratify=y_cls_train
            )
        else:
            # Только один класс - без стратификации
            print("⚠️ ВНИМАНИЕ: Найден только один класс, разделение без стратификации")
            X_train, X_test, y_cls_train, y_cls_test, y_bbox_train, y_bbox_test = train_test_split(
                images, labels, bboxes, test_size=0.2, random_state=42
            )
            
            X_train, X_val, y_cls_train, y_cls_val, y_bbox_train, y_bbox_val = train_test_split(
                X_train, y_cls_train, y_bbox_train, test_size=0.2, random_state=42
            )
        
        print(f"\n📊 Разделение данных:")
        print(f"  Обучение: {X_train.shape[0]} ({Counter(y_cls_train)})")
        print(f"  Валидация: {X_val.shape[0]} ({Counter(y_cls_val)})")
        print(f"  Тест: {X_test.shape[0]} ({Counter(y_cls_test)})")
        
        # Проверяем баланс классов
        unique_classes = np.unique(labels)
        if len(unique_classes) == 1:
            if unique_classes[0] == 1:
                print("\n⚠️ КРИТИЧЕСКАЯ ПРОБЛЕМА: Все изображения имеют кружки!")
                print("   Модель не сможет научиться различать 'есть кружка' и 'нет кружки'")
                print("   Добавьте изображения без кружек в папку или уберите их аннотации")
            else:
                print("\n⚠️ КРИТИЧЕСКАЯ ПРОБЛЕМА: Все изображения БЕЗ кружек!")
                print("   Модель не сможет научиться находить кружки")
            
            print("\n💡 Продолжаем обучение, но результаты будут плохими...")
        
        elif len(unique_classes) == 2:
            class_counts = Counter(labels)
            ratio = min(class_counts.values()) / max(class_counts.values())
            if ratio < 0.3:
                print(f"\n⚠️ ДИСБАЛАНС КЛАССОВ: Соотношение {ratio:.2f}")
                print("   Рекомендуется более сбалансированный датасет")
            else:
                print(f"\n✅ Хороший баланс классов: {class_counts}")
        
        # Создаем модель
        model = create_detection_model()
        
        # Компилируем модель
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={
                'classification': 'binary_crossentropy',
                'bbox_regression': 'mse'
            },
            loss_weights={
                'classification': 1.0,
                'bbox_regression': 1.0
            },
            metrics={
                'classification': ['accuracy'],
                'bbox_regression': ['mae']
            }
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_classification_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='cup_detection_model.h5',
                monitor='val_classification_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Аугментация (осторожная для bbox)
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Параметры обучения
        epochs = 50
        batch_size = 16
        
        print(f"\n🎯 Параметры обучения:")
        print(f"  Эпохи: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: 0.0001")
        
        # Обучение
        print(f"\n{'='*50}")
        print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ")
        print(f"{'='*50}")
        
        history = model.fit(
            X_train,
            {'classification': y_cls_train, 'bbox_regression': y_bbox_train},
            validation_data=(X_val, {'classification': y_cls_val, 'bbox_regression': y_bbox_val}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Загружаем лучшие веса
        try:
            model.load_weights('cup_detection_model.h5')
            print("\n✅ Загружены лучшие веса")
        except:
            print("\n⚠️ Используем текущие веса")
        
        # Оценка модели
        print("\n📊 ОЦЕНКА МОДЕЛИ")
        print("="*40)
        
        # Предсказания на тестовых данных
        predictions = model.predict(X_test, verbose=0)
        cls_pred = predictions[0].squeeze()
        bbox_pred = predictions[1]
        
        # Метрики классификации
        cls_pred_binary = (cls_pred > 0.5).astype(int)
        cls_accuracy = np.mean(cls_pred_binary == y_cls_test)
        
        print(f"Точность классификации: {cls_accuracy:.4f}")
        
        # Метрики bbox (только для изображений с объектами)
        has_object_mask = y_cls_test == 1
        if np.sum(has_object_mask) > 0:
            bbox_mae = mean_absolute_error(
                y_bbox_test[has_object_mask], 
                bbox_pred[has_object_mask]
            )
            bbox_mse = mean_squared_error(
                y_bbox_test[has_object_mask], 
                bbox_pred[has_object_mask]
            )
            
            print(f"Bbox MAE: {bbox_mae:.4f}")
            print(f"Bbox MSE: {bbox_mse:.4f}")
        
        # Визуализация результатов
        visualize_detection_results(X_test[:8], y_cls_test[:8], y_bbox_test[:8], 
                                  cls_pred[:8], bbox_pred[:8])
        
        # График обучения
        plot_training_history(history)
        
        # Сохранение модели
        try:
            model.save('cup_detection_model_final.h5')
            print(f"\n✅ Модель сохранена: cup_detection_model_final.h5")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")
        
        print(f"\n🏆 РЕЗУЛЬТАТЫ:")
        print(f"Точность классификации: {cls_accuracy:.4f}")
        if np.sum(has_object_mask) > 0:
            print(f"Точность bbox (MAE): {bbox_mae:.4f}")
        
        return model, history, cls_accuracy
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0

def visualize_detection_results(images, true_cls, true_bbox, pred_cls, pred_bbox):
    """
    Визуализация результатов детекции
    """
    print("\n🖼️ ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ДЕТЕКЦИИ")
    
    # Денормализация изображений
    def denormalize_imagenet(img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        denorm = img * std + mean
        return np.clip(denorm, 0, 1)
    
    # Функция для рисования bbox
    def draw_bbox(ax, bbox, color, label):
        center_x, center_y, width, height = bbox
        # Конвертируем в corner coordinates
        x = center_x - width / 2
        y = center_y - height / 2
        
        # Масштабируем к размеру изображения (224x224)
        x *= 224
        y *= 224
        width *= 224
        height *= 224
        
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=2, edgecolor=color, 
                               facecolor='none', label=label)
        ax.add_patch(rect)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(8, len(images))):
        img = denormalize_imagenet(images[i])
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Заголовок
        true_label = "Cup" if true_cls[i] > 0.5 else "No Cup"
        pred_label = "Cup" if pred_cls[i] > 0.5 else "No Cup"
        confidence = pred_cls[i]
        
        title = f"True: {true_label}\nPred: {pred_label} ({confidence:.3f})"
        axes[i].set_title(title, fontsize=10)
        
        # Рисуем bbox если есть объект
        if true_cls[i] > 0.5:
            draw_bbox(axes[i], true_bbox[i], 'green', 'True')
        
        if pred_cls[i] > 0.5:
            draw_bbox(axes[i], pred_bbox[i], 'red', 'Pred')
        
        # Легенда
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    График истории обучения
    """
    plt.figure(figsize=(15, 5))
    
    # График точности классификации
    plt.subplot(1, 3, 1)
    plt.plot(history.history['classification_accuracy'], label='Train')
    plt.plot(history.history['val_classification_accuracy'], label='Validation')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # График потерь
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # График bbox MAE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['bbox_regression_mae'], label='Train')
    plt.plot(history.history['val_bbox_regression_mae'], label='Validation')
    plt.title('Bbox MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def test_detection_model(model_path='cup_detection_model_final.h5', 
                        test_image_path=None):
    """
    Тестирование модели детекции на отдельном изображении
    """
    print("🧪 ТЕСТ МОДЕЛИ ДЕТЕКЦИИ")
    print("="*40)
    
    try:
        # Загружаем модель
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Модель загружена: {model_path}")
        
        if test_image_path and os.path.exists(test_image_path):
            # Предобрабатываем изображение
            result = enhanced_preprocessing_detection(test_image_path)
            if result is not None:
                processed_image, scale, x_offset, y_offset = result
                
                # Делаем предсказание
                predictions = model.predict(np.expand_dims(processed_image, axis=0), verbose=0)
                cls_pred = predictions[0][0][0]
                bbox_pred = predictions[1][0]
                
                print(f"\n🎯 Результаты:")
                print(f"  Файл: {os.path.basename(test_image_path)}")
                print(f"  Есть кружка: {'Да' if cls_pred > 0.5 else 'Нет'} (уверенность: {cls_pred:.3f})")
                
                if cls_pred > 0.5:
                    center_x, center_y, width, height = bbox_pred
                    print(f"  Координаты bbox:")
                    print(f"    Центр X: {center_x:.3f}")
                    print(f"    Центр Y: {center_y:.3f}")
                    print(f"    Ширина: {width:.3f}")
                    print(f"    Высота: {height:.3f}")
                    
                    # Визуализируем результат
                    visualize_single_detection(test_image_path, cls_pred, bbox_pred)
            else:
                print("❌ Ошибка обработки изображения")
        else:
            print("⚠️ Путь к изображению не указан или файл не найден")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

def visualize_single_detection(image_path, cls_pred, bbox_pred):
    """
    Визуализация детекции на одном изображении
    """
    try:
        # Загружаем и обрабатываем изображение
        result = enhanced_preprocessing_detection(image_path)
        if result is None:
            return
        
        processed_image, scale, x_offset, y_offset = result
        
        # Денормализация
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        denorm = processed_image * std + mean
        denorm = np.clip(denorm, 0, 1)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(denorm)
        
        # Рисуем bbox если объект обнаружен
        if cls_pred > 0.5:
            center_x, center_y, width, height = bbox_pred
            
            # Конвертируем в corner coordinates
            x = center_x - width / 2
            y = center_y - height / 2
            
            # Масштабируем к размеру изображения
            x *= 224
            y *= 224
            width *= 224
            height *= 224
            
            rect = patches.Rectangle((x, y), width, height, 
                                   linewidth=3, edgecolor='red', 
                                   facecolor='none')
            plt.gca().add_patch(rect)
            
            plt.text(x, y-5, f'Cup: {cls_pred:.3f}', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                    fontsize=12, color='white')
        
        plt.title(f'Detection Result\nConfidence: {cls_pred:.3f}')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Ошибка визуализации: {e}")

if __name__ == "__main__":
    print("🎯 ДЕТЕКЦИЯ КРУЖЕК С КООРДИНАТАМИ BBOX")
    print("="*70)
    print("Возможности:")
    print("✅ Классификация (есть кружка или нет)")
    print("✅ Детекция с координатами bbox")
    print("✅ Transfer Learning с MobileNetV2")
    print("✅ Двойной выход: классификация + регрессия")
    print("✅ Обработка COCO аннотаций")
    print("="*70)
    
    # Запуск обучения
    print(f"\n{'='*50}")
    input("Нажмите Enter для начала обучения...")
    
    model, history, accuracy = train_detection_model()
    
    if accuracy > 0.7:
        print(f"\n{'='*50}")
        print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
        
        # Пример тестирования (замените на реальный путь)
        # test_detection_model('cup_detection_model_final.h5', 'path/to/test/image.jpg')
        
    print(f"\n🏁 ЗАВЕРШЕНО!")
    if accuracy > 0.8:
        print("🎉 Отличный результат!")
    elif accuracy > 0.7:
        print("✅ Хороший результат!")
    else:
        print("❌ Нужны улучшения...")
    
    print("\n💡 Как использовать модель:")
    print("1. Загрузите модель: tf.keras.models.load_model('cup_detection_model_final.h5')")
    print("2. Предобработайте изображение функцией enhanced_preprocessing_detection()")
    print("3. Получите предсказания: model.predict(image)")
    print("4. Результат: [classification_confidence, [center_x, center_y, width, height]]")

