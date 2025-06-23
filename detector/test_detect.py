import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from collections import deque
import os
from datetime import datetime

class CupDetectorCamera:
    def __init__(self, model_path, camera_id=0, confidence_threshold=0.5):
        """
        Инициализация детектора кружек для камеры
        
        Args:
            model_path: путь к обученной модели
            camera_id: ID камеры (обычно 0 для основной)
            confidence_threshold: порог уверенности для детекции
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        
        # Загружаем модель
        print("🔄 Загрузка модели...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Модель загружена успешно!")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise
        
        # Инициализируем камеру
        print(f"🔄 Подключение к камере {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Не удалось подключиться к камере {camera_id}")
            raise RuntimeError(f"Камера {camera_id} недоступна")
        
        # Настройки камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Получаем реальные размеры
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Камера подключена: {self.frame_width}x{self.frame_height}")
        
        # Статистика
        self.fps_counter = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        
        # Флаги
        self.running = False
        self.save_detections = False
        self.detection_count = 0
        
        # Создаем папку для сохранений
        self.save_dir = "cup_detections"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def preprocess_frame(self, frame, target_size=(224, 224)):
        """
        Предобработка кадра для модели
        """
        try:
            # Конвертируем BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            h, w = rgb_frame.shape[:2]
            
            # Вычисляем коэффициент масштабирования
            scale = min(target_size[0] / h, target_size[1] / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Изменяем размер
            resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
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
            
            return normalized, scale, x_offset, y_offset
            
        except Exception as e:
            print(f"Ошибка предобработки: {e}")
            return None
    
    def detect_cups_in_frame(self, frame):
        """
        Детекция кружек в кадре
        """
        start_time = time.time()
        
        # Предобрабатываем кадр
        result = self.preprocess_frame(frame)
        if result is None:
            return None, 0
        
        processed_frame, scale, x_offset, y_offset = result
        
        # Применяем модель
        try:
            predictions = self.model.predict(
                np.expand_dims(processed_frame, axis=0), 
                verbose=0
            )
            
            classification_confidence = predictions[0][0][0]
            bbox_coords = predictions[1][0]
            
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            
            # Формируем результат
            detection_result = {
                'has_cup': classification_confidence > self.confidence_threshold,
                'confidence': float(classification_confidence),
                'bbox': {
                    'center_x': float(bbox_coords[0]),
                    'center_y': float(bbox_coords[1]),
                    'width': float(bbox_coords[2]),
                    'height': float(bbox_coords[3])
                },
                'preprocessing_params': {
                    'scale': scale,
                    'x_offset': x_offset,
                    'y_offset': y_offset
                }
            }
            
            return detection_result, detection_time
            
        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return None, 0
    
    def draw_detection_on_frame(self, frame, detection_result):
        """
        Отрисовывает результаты детекции на кадре
        """
        if detection_result is None:
            return frame
        
        frame_copy = frame.copy()
        h, w = frame_copy.shape[:2]
        
        # Информация в углу экрана
        confidence = detection_result['confidence']
        status_text = f"Cup: {'YES' if detection_result['has_cup'] else 'NO'}"
        confidence_text = f"Conf: {confidence:.3f}"
        
        # Цвет статуса
        status_color = (0, 255, 0) if detection_result['has_cup'] else (0, 0, 255)
        
        # Рисуем фон для текста
        cv2.rectangle(frame_copy, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (10, 10), (300, 80), status_color, 2)
        
        # Рисуем текст
        cv2.putText(frame_copy, status_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame_copy, confidence_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Рисуем bbox если кружка найдена
        if detection_result['has_cup']:
            bbox = detection_result['bbox']
            
            # Конвертируем нормализованные координаты в пиксели
            center_x = bbox['center_x'] * w
            center_y = bbox['center_y'] * h
            width = bbox['width'] * w
            height = bbox['height'] * h
            
            # Corner coordinates
            x1 = int(center_x - width / 2)
            y1 = int(center_y - height / 2)
            x2 = int(center_x + width / 2)
            y2 = int(center_y + height / 2)
            
            # Ограничиваем координаты размерами кадра
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Рисуем bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Рисуем центральную точку
            cv2.circle(frame_copy, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
            
            # Подпись к bbox
            label = f"Cup {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Фон для подписи
            cv2.rectangle(frame_copy, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), 
                         (0, 255, 0), -1)
            
            # Текст подписи
            cv2.putText(frame_copy, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame_copy
    
    def draw_stats_on_frame(self, frame):
        """
        Отрисовывает статистику на кадре
        """
        h, w = frame.shape[:2]
        
        # FPS
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        fps_text = f"FPS: {avg_fps:.1f}"
        
        # Время детекции
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        detection_text = f"Det: {avg_detection_time*1000:.0f}ms"
        
        # Счетчик детекций
        count_text = f"Detections: {self.detection_count}"
        
        # Рисуем статистику в правом углу
        stats_y = h - 100
        cv2.rectangle(frame, (w-200, stats_y-30), (w-10, h-10), (0, 0, 0), -1)
        cv2.rectangle(frame, (w-200, stats_y-30), (w-10, h-10), (255, 255, 255), 1)
        
        cv2.putText(frame, fps_text, (w-190, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, detection_text, (w-190, stats_y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, count_text, (w-190, stats_y+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_instructions(self, frame):
        """
        Отрисовывает инструкции управления
        """
        h, w = frame.shape[:2]
        
        instructions = [
            "Controls:",
            "ESC/Q - Exit",
            "S - Save detection",
            "SPACE - Toggle auto-save",
            "C - Clear counter"
        ]
        
        # Рисуем инструкции в левом нижнем углу
        start_y = h - len(instructions) * 25 - 10
        cv2.rectangle(frame, (10, start_y-10), (200, h-10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, start_y-10), (200, h-10), (255, 255, 255), 1)
        
        for i, instruction in enumerate(instructions):
            y_pos = start_y + i * 20
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, instruction, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Статус автосохранения
        if self.save_detections:
            cv2.putText(frame, "AUTO-SAVE: ON", (15, start_y + len(instructions) * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
    
    def save_detection_image(self, frame, detection_result):
        """
        Сохраняет кадр с детекцией
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if detection_result and detection_result['has_cup']:
            conf = detection_result['confidence']
            filename = f"cup_detected_{timestamp}_conf{conf:.3f}.jpg"
        else:
            filename = f"no_cup_{timestamp}.jpg"
        
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"💾 Сохранено: {filename}")
        
        return filepath
    
    def run(self):
        """
        Основной цикл детекции
        """
        print("\n🚀 ЗАПУСК REAL-TIME ДЕТЕКЦИИ КРУЖЕК")
        print("=" * 50)
        print("ESC или Q - выход")
        print("S - сохранить текущий кадр")
        print("SPACE - вкл/выкл автосохранение детекций")
        print("C - сбросить счетчик")
        print("=" * 50)
        
        self.running = True
        
        try:
            while self.running:
                frame_start_time = time.time()
                
                # Захватываем кадр
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Ошибка захвата кадра")
                    break
                
                # Детекция
                detection_result, detection_time = self.detect_cups_in_frame(frame)
                
                # Подсчет статистики
                if detection_result and detection_result['has_cup']:
                    self.detection_count += 1
                
                # Отрисовка результатов
                frame_with_detection = self.draw_detection_on_frame(frame, detection_result)
                frame_with_stats = self.draw_stats_on_frame(frame_with_detection)
                final_frame = self.draw_instructions(frame_with_stats)
                
                # Автосохранение
                if (self.save_detections and detection_result and 
                    detection_result['has_cup'] and 
                    detection_result['confidence'] > 0.8):
                    self.save_detection_image(final_frame, detection_result)
                
                # Показываем кадр
                cv2.imshow('Cup Detection Camera', final_frame)
                
                # Подсчет FPS
                frame_time = time.time() - frame_start_time
                self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
                
                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC или Q
                    break
                elif key == ord('s'):  # S - сохранить
                    self.save_detection_image(final_frame, detection_result)
                elif key == ord(' '):  # SPACE - автосохранение
                    self.save_detections = not self.save_detections
                    status = "ON" if self.save_detections else "OFF"
                    print(f"🔄 Автосохранение: {status}")
                elif key == ord('c'):  # C - сбросить счетчик
                    self.detection_count = 0
                    print("🔄 Счетчик сброшен")
                
        except KeyboardInterrupt:
            print("\n⛔ Прервано пользователем")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Очистка ресурсов
        """
        print("\n🧹 Очистка ресурсов...")
        self.running = False
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Статистика сессии
        print(f"\n📊 СТАТИСТИКА СЕССИИ:")
        print(f"   Всего детекций кружек: {self.detection_count}")
        print(f"   Средний FPS: {np.mean(self.fps_counter):.1f}")
        print(f"   Среднее время детекции: {np.mean(self.detection_times)*1000:.0f}ms")
        print(f"   Сохранения в папке: {self.save_dir}")
        print("✅ Завершено!")

# ===== Функции для быстрого запуска =====

def test_camera_connection(camera_id=0):
    """
    Тестирует подключение к камере
    """
    print(f"🔍 Тестирование камеры {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Камера {camera_id} недоступна")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Не удалось захватить кадр с камеры {camera_id}")
        cap.release()
        return False
    
    h, w = frame.shape[:2]
    print(f"✅ Камера {camera_id} работает: {w}x{h}")
    
    cap.release()
    return True

def list_available_cameras(max_cameras=5):
    """
    Находит доступные камеры
    """
    print("🔍 Поиск доступных камер...")
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                available_cameras.append((i, w, h))
                print(f"✅ Камера {i}: {w}x{h}")
        cap.release()
    
    if not available_cameras:
        print("❌ Камеры не найдены")
    
    return available_cameras

def run_cup_detection_camera(model_path="cup_detection_model_final.h5", 
                           camera_id=0, 
                           confidence_threshold=0.5):
    """
    Запускает детекцию кружек через камеру
    """
    try:
        # Проверяем наличие модели
        if not os.path.exists(model_path):
            print(f"❌ Модель не найдена: {model_path}")
            return
        
        # Тестируем камеру
        if not test_camera_connection(camera_id):
            print("💡 Попробуйте другой camera_id или проверьте подключение камеры")
            return
        
        # Создаем и запускаем детектор
        detector = CupDetectorCamera(
            model_path=model_path,
            camera_id=camera_id,
            confidence_threshold=confidence_threshold
        )
        
        detector.run()
        
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        import traceback
        traceback.print_exc()

# ===== MAIN =====

if __name__ == "__main__":
    print("🎯 REAL-TIME ДЕТЕКЦИЯ КРУЖЕК ЧЕРЕЗ USB КАМЕРУ")
    print("=" * 60)
    
    # Поиск камер
    cameras = list_available_cameras()
    
    if not cameras:
        print("❌ Камеры не найдены. Проверьте подключение USB камеры.")
        exit(1)
    
    # Запуск детекции
    print(f"\n🚀 Запуск детекции на камере 0...")
    run_cup_detection_camera(
        model_path="cup_detection_model_final.h5",
        camera_id=0,
        confidence_threshold=0.3
    )