import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Фиксируем семена
np.random.seed(42)
tf.random.set_seed(42)


def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Создает модель на основе MobileNetV2 (transfer learning)
    """
    print("\n🧠 СОЗДАНИЕ МОДЕЛИ С TRANSFER LEARNING")
    print("="*50)
    
    # Загружаем предобученную модель
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Замораживаем веса базовой модели
    base_model.trainable = False
    
    # Добавляем наши слои
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    # Компилируем с низким learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Базовая модель: MobileNetV2")
    print(f"Общее количество параметров: {model.count_params():,}")
    print(f"Обучаемых параметров: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    return model

def enhanced_preprocessing(image_path, target_size=(224, 224)):
    """
    Улучшенная предобработка с нормализацией для ImageNet
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
        
        # Вычисляем новые размеры
        if h > w:
            new_h = target_size[0]
            new_w = int(w * target_size[0] / h)
        else:
            new_w = target_size[1]
            new_h = int(h * target_size[1] / w)
        
        # Изменяем размер
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Создаем финальное изображение с padding
        final_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # Центрируем изображение
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2
        final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Нормализация для ImageNet (важно для transfer learning!)
        normalized = final_image.astype("float32") / 255.0
        
        # ImageNet нормализация
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        return normalized
        
    except Exception as e:
        print(f"Ошибка обработки {image_path}: {e}")
        return None

def create_contrast_enhanced_dataset(images_dir="train", annotations_file="train/_annotations.coco.json", 
                                   target_size=(224, 224), samples_per_class=150):
    """
    Создает контрастный датасет с улучшенной предобработкой
    """
    print("🎯 СОЗДАНИЕ КОНТРАСТНОГО ДАТАСЕТА")
    print("="*50)
    
    # Глубокий анализ данных
    files_with_cups, files_without_cups = deep_data_analysis(images_dir, annotations_file)
    
    print(f"\nИспользуем по {samples_per_class} образцов каждого класса")
    
    # Случайно выбираем образцы
    cup_files = list(files_with_cups)[:samples_per_class]
    no_cup_files = list(files_without_cups)[:samples_per_class]
    
    data = []
    labels = []
    filenames = []
    
    print(f"\n📥 Загрузка изображений с кружками...")
    successful_cups = 0
    for i, filename in enumerate(cup_files):
        if i % 25 == 0:
            print(f"  Обработано: {i}/{len(cup_files)}")
        
        image_path = os.path.join(images_dir, filename)
        if os.path.exists(image_path):
            processed_image = enhanced_preprocessing(image_path, target_size)
            if processed_image is not None:
                data.append(processed_image)
                labels.append(1)  # cup
                filenames.append(filename)
                successful_cups += 1
    
    print(f"📥 Загрузка изображений без кружек...")
    successful_no_cups = 0
    for i, filename in enumerate(no_cup_files):
        if i % 25 == 0:
            print(f"  Обработано: {i}/{len(no_cup_files)}")
        
        image_path = os.path.join(images_dir, filename)
        if os.path.exists(image_path):
            processed_image = enhanced_preprocessing(image_path, target_size)
            if processed_image is not None:
                data.append(processed_image)
                labels.append(0)  # no cup
                filenames.append(filename)
                successful_no_cups += 1
    
    print(f"\n✅ Успешно загружено:")
    print(f"  С кружками: {successful_cups}")
    print(f"  Без кружек: {successful_no_cups}")
    print(f"  Итого: {len(data)} изображений")
    
    if len(data) == 0:
        raise ValueError("Не удалось загрузить ни одного изображения!")
    
    # Конвертируем в numpy массивы
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    print(f"\nРаспределение классов: {Counter(labels)}")
    
    # Статистика загруженных данных
    print(f"\n📊 Статистика данных:")
    print(f"  Форма: {data.shape}")
    print(f"  Диапазон: [{data.min():.3f}, {data.max():.3f}]")
    print(f"  Среднее: {data.mean():.3f}")
    print(f"  Стд. отклонение: {data.std():.3f}")
    
    # Анализ различий между классами
    cup_data = data[labels == 1]
    no_cup_data = data[labels == 0]
    
    if len(cup_data) > 0 and len(no_cup_data) > 0:
        cup_mean = np.mean(cup_data)
        no_cup_mean = np.mean(no_cup_data)
        difference = abs(cup_mean - no_cup_mean)
        
        print(f"\n🔍 Различия между классами:")
        print(f"  Среднее для кружек: {cup_mean:.4f}")
        print(f"  Среднее для не-кружек: {no_cup_mean:.4f}")
        print(f"  Абсолютная разность: {difference:.4f}")
        
        if difference > 0.1:
            print("✅ Классы хорошо различимы!")
        elif difference > 0.05:
            print("⚠️ Классы умеренно различимы")
        else:
            print("❌ Классы плохо различимы!")
    
    return data, labels, filenames

def show_sample_images_detailed(data, labels, filenames, num_samples=8):
    """
    Детальная визуализация образцов с анализом
    """
    print(f"\n🖼️ ДЕТАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ОБРАЗЦОВ")
    print("="*50)
    
    # Денормализация для отображения (обратная ImageNet нормализация)
    def denormalize_imagenet(img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        denorm = img * std + mean
        return np.clip(denorm, 0, 1)
    
    cup_indices = np.where(labels == 1)[0]
    no_cup_indices = np.where(labels == 0)[0]
    
    fig, axes = plt.subplots(3, num_samples//2, figsize=(16, 8))
    
    # Показываем кружки
    print("Анализ изображений с кружками:")
    for i in range(num_samples//2):
        if i < len(cup_indices):
            idx = cup_indices[i]
            img = denormalize_imagenet(data[idx])
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Cup: {filenames[idx][:12]}...\nMean: {np.mean(data[idx]):.3f}', fontsize=8)
            axes[0, i].axis('off')
            
            print(f"  {i+1}. {filenames[idx]}: mean={np.mean(data[idx]):.3f}, std={np.std(data[idx]):.3f}")
    
    # Показываем не-кружки
    print("\nАнализ изображений без кружек:")
    for i in range(num_samples//2):
        if i < len(no_cup_indices):
            idx = no_cup_indices[i]
            img = denormalize_imagenet(data[idx])
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'No Cup: {filenames[idx][:12]}...\nMean: {np.mean(data[idx]):.3f}', fontsize=8)
            axes[1, i].axis('off')
            
            print(f"  {i+1}. {filenames[idx]}: mean={np.mean(data[idx]):.3f}, std={np.std(data[idx]):.3f}")
    
    # Показываем разности
    for i in range(num_samples//2):
        if i < len(cup_indices) and i < len(no_cup_indices):
            cup_idx = cup_indices[i]
            no_cup_idx = no_cup_indices[i]
            
            diff = np.abs(data[cup_idx] - data[no_cup_idx])
            axes[2, i].imshow(diff, cmap='hot')
            axes[2, i].set_title(f'Diff\nMax: {np.max(diff):.3f}', fontsize=8)
            axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def train_advanced_classifier():
    """
    Обучение продвинутого классификатора с transfer learning
    """
    print("🚀 ПРОДВИНУТОЕ ОБУЧЕНИЕ С TRANSFER LEARNING")
    print("="*70)
    
    try:
        # Создаем контрастный датасет
        data, labels, filenames = create_contrast_enhanced_dataset(
            images_dir="train",
            annotations_file="train/_annotations.coco.json",
            target_size=(224, 224),
            samples_per_class=150
        )
        
        if len(data) < 10:
            raise ValueError("Недостаточно данных для обучения!")
        
        # Детальная визуализация
        show_sample_images_detailed(data, labels, filenames)
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"\n📊 Разделение данных:")
        print(f"  Обучение: {X_train.shape[0]} ({Counter(y_train)})")
        print(f"  Валидация: {X_val.shape[0]} ({Counter(y_val)})")
        print(f"  Тест: {X_test.shape[0]} ({Counter(y_test)})")
        
        # Создаем модель с transfer learning
        model = create_transfer_learning_model()
        
        # Агрессивная аугментация для малого датасета
        train_datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Улучшенные callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.005
            ),
            ModelCheckpoint(
                filepath='advanced_cup_classifier.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=8,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        # Параметры обучения
        epochs = 100
        batch_size = 16  # Маленький batch для стабильности
        
        print(f"\n🎯 Параметры обучения:")
        print(f"  Модель: MobileNetV2 + Transfer Learning")
        print(f"  Эпохи: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: 0.0001")
        print(f"  Аугментация: агрессивная")
        
        # Обучение
        print(f"\n{'='*50}")
        print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ")
        print(f"{'='*50}")
        
        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=max(1, len(X_train) // batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Загружаем лучшие веса
        try:
            model.load_weights('advanced_cup_classifier.h5')
            print("\n✅ Загружены лучшие веса")
        except:
            print("\n⚠️ Используем текущие веса")
        
        # Fine-tuning: размораживаем последние слои базовой модели
        print("\n🔥 FINE-TUNING: Размораживаем последние слои")
        history_fine = None  # Инициализируем переменную
        
        try:
            base_model = model.layers[0]
            base_model.trainable = True
            
            # Замораживаем все слои кроме последних 20
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            # Перекомпилируем с очень низким learning rate
            model.compile(
                optimizer=Adam(learning_rate=0.00001),  # Очень низкий LR для fine-tuning
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"Обучаемых параметров после разморозки: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
            
            # Продолжаем обучение с fine-tuning
            print("\n🔥 Продолжаем обучение с fine-tuning...")
            
            history_fine = model.fit(
                train_datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=max(1, len(X_train) // batch_size),
                validation_data=(X_val, y_val),
                epochs=25,  # Меньше эпох для fine-tuning
                callbacks=callbacks,
                verbose=1,
                initial_epoch=len(history.history['accuracy'])
            )
            
            print("✅ Fine-tuning завершен успешно")
            
        except Exception as e:
            print(f"⚠️ Ошибка при fine-tuning: {e}")
            print("Продолжаем с базовой моделью...")
            history_fine = None
        
        # Оценка финальной модели
        print("\n📊 ФИНАЛЬНАЯ ОЦЕНКА")
        print("="*40)
        
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Точность на обучении: {train_acc:.4f}")
        print(f"Точность на валидации: {val_acc:.4f}")
        print(f"Точность на тесте: {test_acc:.4f}")
        
        # Детальный анализ предсказаний
        test_predictions = model.predict(X_test, verbose=0)
        test_pred_classes = test_predictions.argmax(axis=1)
        
        # Анализ уверенности
        confidences = test_predictions.max(axis=1)
        print(f"\n🎯 Анализ уверенности:")
        print(f"  Средняя: {np.mean(confidences):.3f}")
        print(f"  Медиана: {np.median(confidences):.3f}")
        print(f"  Минимальная: {np.min(confidences):.3f}")
        print(f"  Максимальная: {np.max(confidences):.3f}")
        
        uncertain = np.sum(confidences < 0.7)
        print(f"  Неуверенных предсказаний (<0.7): {uncertain}/{len(confidences)} ({uncertain/len(confidences)*100:.1f}%)")
        
        # Отчет по классификации
        print(f"\n📋 Детальный отчет:")
        class_names = ['no_cup', 'cup']
        report = classification_report(y_test, test_pred_classes, 
                                     target_names=class_names, digits=4)
        print(report)
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, test_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Количество'})
        plt.title(f'Матрица ошибок\nТочность: {test_acc:.3f}, Средняя уверенность: {np.mean(confidences):.3f}')
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Истинный класс')
        plt.show()
        
        # График обучения
        plt.figure(figsize=(15, 5))
        
        # Объединяем историю обучения (безопасно)
        all_accuracy = history.history['accuracy'][:]
        all_val_accuracy = history.history['val_accuracy'][:]
        all_loss = history.history['loss'][:]
        all_val_loss = history.history['val_loss'][:]
        
        # Добавляем fine-tuning данные если они есть
        if history_fine is not None and hasattr(history_fine, 'history'):
            all_accuracy.extend(history_fine.history.get('accuracy', []))
            all_val_accuracy.extend(history_fine.history.get('val_accuracy', []))
            all_loss.extend(history_fine.history.get('loss', []))
            all_val_loss.extend(history_fine.history.get('val_loss', []))
            fine_tuning_start = len(history.history['accuracy'])
        else:
            fine_tuning_start = None
        
        plt.subplot(1, 3, 1)
        plt.plot(all_accuracy, label='Обучение')
        plt.plot(all_val_accuracy, label='Валидация')
        if fine_tuning_start is not None:
            plt.axvline(x=fine_tuning_start, color='red', linestyle='--', alpha=0.7, label='Fine-tuning start')
        plt.title('Точность')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(all_loss, label='Обучение')
        plt.plot(all_val_loss, label='Валидация')
        if fine_tuning_start is not None:
            plt.axvline(x=fine_tuning_start, color='red', linestyle='--', alpha=0.7, label='Fine-tuning start')
        plt.title('Потери')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(confidences), color='red', linestyle='--', label=f'Среднее: {np.mean(confidences):.3f}')
        plt.title('Распределение уверенности')
        plt.xlabel('Уверенность')
        plt.ylabel('Количество')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Сохранение
        try:
            model.save('advanced_cup_classifier_final.h5')
            print(f"\n✅ Модель сохранена: advanced_cup_classifier_final.h5")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")
        
        # Финальная оценка
        print(f"\n" + "="*60)
        print("🏆 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print("="*60)
        
        if test_acc > 0.9:
            status = "ОТЛИЧНО ✅"
        elif test_acc > 0.8:
            status = "ХОРОШО ✅"
        elif test_acc > 0.7:
            status = "УДОВЛЕТВОРИТЕЛЬНО ⚠️"
        else:
            status = "НЕУДОВЛЕТВОРИТЕЛЬНО ❌"
        
        print(f"Статус: {status}")
        print(f"Тестовая точность: {test_acc:.4f}")
        print(f"Средняя уверенность: {np.mean(confidences):.3f}")
        
        if test_acc > 0.8 and np.mean(confidences) > 0.8:
            print(f"\n🎉 ОТЛИЧНЫЙ РЕЗУЛЬТАТ! Модель готова к использованию")
            print("📁 Используйте файл: advanced_cup_classifier_final.h5")
        elif test_acc > 0.7:
            print(f"\n✅ Хороший результат. Модель работает.")
        else:
            print(f"\n💡 Рекомендации:")
            print("   - Проверить качество данных")
            print("   - Собрать больше контрастных примеров")
            print("   - Попробовать другую архитектуру")
        
        return model, history, test_acc
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0

def quick_test_on_sample(model_path='advanced_cup_classifier_final.h5', 
                        images_dir="train", num_test_images=10):
    """
    Быстрый тест модели на случайных изображениях
    """
    print("🧪 БЫСТРЫЙ ТЕСТ МОДЕЛИ")
    print("="*40)
    
    try:
        # Загружаем модель
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Модель загружена: {model_path}")
        
        # Получаем случайные файлы
        all_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_files.extend(glob(os.path.join(images_dir, ext)))
        
        test_files = np.random.choice(all_files, min(num_test_images, len(all_files)), replace=False)
        
        print(f"\n🎯 Тестируем на {len(test_files)} случайных изображениях:")
        
        for i, file_path in enumerate(test_files):
            filename = os.path.basename(file_path)
            
            # Предобрабатываем изображение
            processed = enhanced_preprocessing(file_path)
            if processed is not None:
                # Делаем предсказание
                prediction = model.predict(np.expand_dims(processed, axis=0), verbose=0)
                predicted_class = prediction.argmax()
                confidence = prediction.max()
                
                class_name = "кружка" if predicted_class == 1 else "не кружка"
                print(f"  {i+1}. {filename[:30]:30} -> {class_name:10} (уверенность: {confidence:.3f})")
            else:
                print(f"  {i+1}. {filename[:30]:30} -> ОШИБКА ОБРАБОТКИ")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

def diagnose_training_problems():
    """
    Диагностирует возможные проблемы с обучением
    """
    print("🔧 ДИАГНОСТИКА ПРОБЛЕМ ОБУЧЕНИЯ")
    print("="*50)
    
    print("Возможные причины плохого обучения:")
    print("\n1. 📊 Проблемы с данными:")
    print("   - Неправильное разделение классов")
    print("   - Слишком похожие изображения между классами")
    print("   - Малое количество данных")
    print("   - Несбалансированные классы")
    
    print("\n2. 🧠 Проблемы с моделью:")
    print("   - Слишком сложная архитектура для малых данных")
    print("   - Неподходящий learning rate")
    print("   - Переобучение")
    
    print("\n3. ⚙️ Проблемы с предобработкой:")
    print("   - Неправильная нормализация")
    print("   - Потеря важной информации при resize")
    print("   - Неподходящая аугментация")
    
    print("\n💡 Рекомендации:")
    print("   1. Используйте transfer learning (как в этом коде)")
    print("   2. Проверьте качество разделения данных")
    print("   3. Увеличьте количество данных")
    print("   4. Попробуйте разные архитектуры")
    print("   5. Настройте гиперпараметры")

if __name__ == "__main__":
    print("🚀 ПРОДВИНУТЫЙ КЛАССИФИКАТОР КРУЖЕК")
    print("="*70)
    print("Ключевые улучшения:")
    print("✅ Transfer Learning с MobileNetV2")
    print("✅ Правильная ImageNet нормализация")
    print("✅ Глубокий анализ данных")
    print("✅ Fine-tuning для лучших результатов")
    print("✅ Агрессивная аугментация")
    print("✅ Детальная диагностика")
    print("="*70)
    
    # Сначала диагностируем проблемы
    diagnose_training_problems()
    
    print(f"\n{'='*50}")
    input("Нажмите Enter для начала обучения...")
    
    # Запускаем обучение
    model, history, accuracy = train_advanced_classifier()
    
    # Если модель обучилась, тестируем её
    if accuracy > 0.6:
        print(f"\n{'='*50}")
        print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
        quick_test_on_sample()
    
    print(f"\n🏁 ЗАВЕРШЕНО!")
    if accuracy > 0.8:
        print("🎉 Отличный результат!")
    elif accuracy > 0.7:
        print("✅ Хороший результат!")
    else:
        print("❌ Нужны улучшения...")

