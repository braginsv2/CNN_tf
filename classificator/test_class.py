import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import threading
import queue
from collections import deque
import argparse
import os

class RealTimeCupDetector:
    def __init__(self, model_path='advanced_cup_classifier.h5', target_size=(224, 224)):
        """
        Инициализация детектора кружек в реальном времени
        """
        self.target_size = target_size
        self.model = None
        self.cap = None
        
        # Буфер для сглаживания предсказаний
        self.prediction_buffer = deque(maxlen=10)
        
        # Статистика производительности
        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        
        # Настройки отображения
        self.confidence_threshold = 0.7
        self.show_preprocessing = False
        
        # Загружаем модель
        self.load_model(model_path)
        
        print("🎥 ДЕТЕКТОР КРУЖЕК В РЕАЛЬНОМ ВРЕМЕНИ")
        print("="*50)
        print("Управление:")
        print("  'q' - выход")
        print("  'p' - показать/скрыть предобработку")
        print("  's' - сохранить кадр")
        print("  '+'/'-' - изменить порог уверенности")
        print("  'r' - сброс статистики")
        print("="*50)
    
    def load_model(self, model_path):
        """
        Загрузка модели классификации
        """
        try:
            print(f"🤖 Загружаем модель: {model_path}")
            self.model = load_model(model_path)
            print(f"✅ Модель загружена! Параметров: {self.model.count_params():,}")
            
            # Прогреваем модель
            dummy_input = np.zeros((1, *self.target_size, 3), dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)
            print("🔥 Модель прогрета для быстрой работы")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
    
    def preprocess_frame(self, frame):
        """
        Предобработка кадра для модели (та же, что при обучении)
        """
        try:
            # Конвертируем BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Изменяем размер с сохранением пропорций
            h, w = rgb_frame.shape[:2]
            
            if h > w:
                new_h = self.target_size[0]
                new_w = int(w * self.target_size[0] / h)
            else:
                new_w = self.target_size[1]
                new_h = int(h * self.target_size[1] / w)
            
            # Изменяем размер
            resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Создаем финальное изображение с padding
            final_image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            
            # Центрируем изображение
            y_offset = (self.target_size[0] - new_h) // 2
            x_offset = (self.target_size[1] - new_w) // 2
            final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # ImageNet нормализация
            normalized = final_image.astype("float32") / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            return normalized, final_image
            
        except Exception as e:
            print(f"Ошибка предобработки: {e}")
            return None, None
    
    def predict_cup(self, frame):
        """
        Предсказание наличия кружки в кадре
        """
        if self.model is None:
            return None, 0.0
        
        processed_frame, display_frame = self.preprocess_frame(frame)
        if processed_frame is None:
            return None, 0.0
        
        # Предсказание
        start_time = time.time()
        prediction = self.model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)
        inference_time = time.time() - start_time
        
        predicted_class = prediction.argmax()
        confidence = prediction.max()
        
        # Добавляем в буфер для сглаживания
        self.prediction_buffer.append((predicted_class, confidence))
        
        # Сглаженное предсказание
        if len(self.prediction_buffer) >= 3:
            recent_predictions = list(self.prediction_buffer)[-5:]  # Последние 5 кадров
            avg_confidence = np.mean([conf for _, conf in recent_predictions])
            most_common_class = max(set([cls for cls, _ in recent_predictions]), 
                                  key=[cls for cls, _ in recent_predictions].count)
            
            class_name = "Cup" if most_common_class == 1 else "No Cup"
            return class_name, avg_confidence, inference_time, display_frame
        else:
            class_name = "Cup" if predicted_class == 1 else "No Cup"
            return class_name, confidence, inference_time, display_frame
    
    def draw_results(self, frame, prediction, confidence, fps, inference_time):
        """
        Отрисовка результатов на кадре
        """
        height, width = frame.shape[:2]
        
        # Цвета для разных состояний
        if prediction == "Cup":
            if confidence > self.confidence_threshold:
                color = (0, 255, 0)  # Зеленый - уверенно кружка
                status = "CUP DETECTED"
            else:
                color = (0, 255, 255)  # Желтый - возможно кружка
                status = "CUP (LOW CONF)"
        else:
            if confidence > self.confidence_threshold:
                color = (0, 0, 255)  # Красный - уверенно не кружка
                status = "NO CUP"
            else:
                color = (128, 128, 128)  # Серый - неопределенность
                status = "UNCERTAIN"
        
        # Основная информация
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Статус (большими буквами)
        cv2.putText(frame, status, (20, 50), font, 1.5, color, 3)
        
        # Детальная информация
        info_lines = [
            f"Confidence: {confidence:.3f}",
            f"FPS: {fps:.1f}",
            f"Inference: {inference_time*1000:.1f}ms",
            f"Threshold: {self.confidence_threshold:.2f}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 100 + i * 30
            cv2.putText(frame, line, (20, y_pos), font, 0.7, (255, 255, 255), 2)
        
        # Индикатор уверенности (полоса)
        bar_width = 300
        bar_height = 20
        bar_x = width - bar_width - 20
        bar_y = 30
        
        # Фон полосы
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Заполнение полосы
        fill_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # Рамка полосы
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Порог уверенности (линия)
        threshold_x = bar_x + int(bar_width * self.confidence_threshold)
        cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), 
                (255, 255, 0), 2)
        
        # Большой индикатор в центре для ярких обнаружений
        if prediction == "Cup" and confidence > self.confidence_threshold:
            center_x, center_y = width // 2, height // 2
            cv2.circle(frame, (center_x, center_y), 100, (0, 255, 0), 5)
            cv2.putText(frame, "☕", (center_x - 30, center_y + 15), 
                       font, 2, (0, 255, 0), 3)
        
        return frame
    
    def run_webcam_detection(self, camera_id=0):
        """
        Запуск детекции через веб-камеру
        """
        if self.model is None:
            print("❌ Модель не загружена!")
            return
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"❌ Не удалось открыть камеру {camera_id}")
            return
        
        # Настройки камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✅ Камера {camera_id} инициализирована")
        print("🎬 Начинаем детекцию...")
        
        saved_frames = 0
        
        while True:
            start_time = time.time()
            
            # Читаем кадр
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Не удалось прочитать кадр")
                break
            
            self.frame_count += 1
            
            # Предсказание
            result = self.predict_cup(frame)
            if result is None:
                continue
            
            if len(result) == 4:
                prediction, confidence, inference_time, preprocessed_display = result
            else:
                prediction, confidence, inference_time = result
                preprocessed_display = None
            
            # Вычисляем FPS
            frame_time = time.time() - start_time
            self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
            current_fps = np.mean(self.fps_counter)
            
            # Отрисовываем результаты
            display_frame = self.draw_results(frame, prediction, confidence, 
                                            current_fps, inference_time)
            
            # Показываем основное окно
            cv2.imshow('Cup Detector - Main', display_frame)
            
            # Показываем предобработку (если включено)
            if self.show_preprocessing and preprocessed_display is not None:
                # Конвертируем для отображения
                display_prep = cv2.cvtColor(preprocessed_display, cv2.COLOR_RGB2BGR)
                display_prep = cv2.resize(display_prep, (300, 300))
                cv2.imshow('Cup Detector - Preprocessing', display_prep)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 Выход...")
                break
            elif key == ord('p'):
                self.show_preprocessing = not self.show_preprocessing
                if not self.show_preprocessing:
                    cv2.destroyWindow('Cup Detector - Preprocessing')
                print(f"Предобработка: {'включена' if self.show_preprocessing else 'выключена'}")
            elif key == ord('s'):
                # Сохранение кадра
                filename = f"cup_detection_frame_{saved_frames:04d}.jpg"
                cv2.imwrite(filename, display_frame)
                saved_frames += 1
                print(f"💾 Кадр сохранен: {filename}")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.05)
                print(f"Порог уверенности: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.01, self.confidence_threshold - 0.05)
                print(f"Порог уверенности: {self.confidence_threshold:.2f}")
            elif key == ord('r'):
                self.prediction_buffer.clear()
                self.fps_counter.clear()
                self.frame_count = 0
                print("🔄 Статистика сброшена")
        
        # Освобождаем ресурсы
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"📊 Статистика сессии:")
        print(f"  Обработано кадров: {self.frame_count}")
        print(f"  Средний FPS: {np.mean(self.fps_counter):.1f}")
        print(f"  Сохранено кадров: {saved_frames}")
    
    def run_video_file_detection(self, video_path, output_path=None):
        """
        Запуск детекции на видеофайле
        """
        if self.model is None:
            print("❌ Модель не загружена!")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Видеофайл не найден: {video_path}")
            return
        
        # Открываем видеофайл
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видеофайл: {video_path}")
            return
        
        # Получаем информацию о видео
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Видеофайл: {video_path}")
        print(f"  Кадров: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Размер: {width}x{height}")
        
        # Настраиваем вывод (если нужно)
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Результат будет сохранен в: {output_path}")
        
        frame_num = 0
        cup_detections = 0
        
        print("🎬 Начинаем обработку видео...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Показываем прогресс
            if frame_num % 30 == 0:
                progress = frame_num / total_frames * 100
                print(f"Прогресс: {progress:.1f}% ({frame_num}/{total_frames})")
            
            # Предсказание
            result = self.predict_cup(frame)
            if result is None:
                continue
            
            prediction, confidence, inference_time = result[:3]
            
            # Считаем обнаружения
            if prediction == "Cup" and confidence > self.confidence_threshold:
                cup_detections += 1
            
            # Отрисовываем результаты
            current_fps = fps  # Для видеофайла используем оригинальный FPS
            display_frame = self.draw_results(frame, prediction, confidence, 
                                            current_fps, inference_time)
            
            # Показываем кадр
            cv2.imshow('Cup Detector - Video', display_frame)
            
            # Сохраняем (если нужно)
            if writer:
                writer.write(display_frame)
            
            # Проверяем клавиши
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("⏹️ Остановлено пользователем")
                break
        
        # Освобождаем ресурсы
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Статистика
        detection_rate = cup_detections / frame_num * 100 if frame_num > 0 else 0
        print(f"\n📊 Статистика обработки:")
        print(f"  Обработано кадров: {frame_num}/{total_frames}")
        print(f"  Кадров с кружками: {cup_detections}")
        print(f"  Процент обнаружений: {detection_rate:.1f}%")
        if output_path and writer:
            print(f"✅ Результат сохранен: {output_path}")

def main():
    """
    Основная функция с аргументами командной строки
    """
    parser = argparse.ArgumentParser(description='Детектор кружек в реальном времени')
    parser.add_argument('--model', '-m', default='advanced_cup_classifier.h5', 
                       help='Путь к модели')
    parser.add_argument('--camera', '-c', type=int, default=0, 
                       help='ID камеры (по умолчанию 0)')
    parser.add_argument('--video', '-v', type=str, 
                       help='Путь к видеофайлу (вместо камеры)')
    parser.add_argument('--output', '-o', type=str, 
                       help='Путь для сохранения результата (только для видео)')
    parser.add_argument('--threshold', '-t', type=float, default=0.7, 
                       help='Порог уверенности (по умолчанию 0.7)')
    
    args = parser.parse_args()
    
    # Создаем детектор
    detector = RealTimeCupDetector(args.model)
    detector.confidence_threshold = args.threshold
    
    if detector.model is None:
        print("❌ Не удалось загрузить модель. Проверьте путь к файлу.")
        return
    
    try:
        if args.video:
            # Обработка видеофайла
            detector.run_video_file_detection(args.video, args.output)
        else:
            # Обработка с камеры
            detector.run_webcam_detection(args.camera)
    except KeyboardInterrupt:
        print("\n⏹️ Остановлено пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    # Можно запускать и без аргументов для быстрого тестирования
    if len(os.sys.argv) == 1:
        # Простой запуск с камерой
        detector = RealTimeCupDetector()
        if detector.model is not None:
            try:
                detector.run_webcam_detection()
            except KeyboardInterrupt:
                print("\n⏹️ Остановлено")
        else:
            print("❌ Модель не найдена. Убедитесь, что файл 'advanced_cup_classifier.h5' существует.")
    else:
        main()