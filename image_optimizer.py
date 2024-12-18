import io
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import tensorflow_hub as hub

class ImageOptimizer:
    def __init__(self):
        self.object_detection_model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")
        print("Model loaded successfully.")

    def load_image(self, image_path):
        image = Image.open(image_path)
        return image

    def detect_objects(self, image):
        image_np = np.array(image)
        converted_image = tf.convert_to_tensor(image_np, dtype=tf.uint8)
        outputs = self.object_detection_model(converted_image)

        return outputs['detection_class_entities'], outputs['detection_boxes'], outputs['detection_scores']

    def enhance_image(self, image, contrast_factor=1.2, sharpness_factor=1.1):
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness_factor)

        return image

    def process_image(self, image_path):
        image = self.load_image(image_path)
        print("Image loaded.")

        # Object Detection
        objects, boxes, scores = self.detect_objects(image)
        print("Objects detected:", objects.numpy())

        # Image Enhancement
        enhanced_image = self.enhance_image(image)
        print("Image enhanced.")

        return enhanced_image, objects, boxes, scores

def save_image(image, output_path):
    image.save(output_path, format="JPEG")

if __name__ == "__main__":
    optimizer = ImageOptimizer()
    enhanced_image, objects, boxes, scores = optimizer.process_image('input.jpg')
    save_image(enhanced_image, 'output.jpg')
    print("Processed image saved as 'output.jpg'.")
