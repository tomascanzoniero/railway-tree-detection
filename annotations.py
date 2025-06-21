import cv2
import numpy as np
import os
import glob
from pathlib import Path
import re

def natural_sort_key(s):
    """
    A key function for natural sorting. Extracts numbers from a string.
    e.g. 'output_10.png' -> 10
    """
    # Use a regular expression to find all sequences of digits in the filename
    # os.path.basename(s) ensures we only look at the filename, not the folder path
    numbers = re.findall(r'\d+', os.path.basename(s))
    # If numbers are found, return the first one as an integer.
    # Otherwise, return 0 so non-numbered files are grouped together.
    return int(numbers[0]) if numbers else 0


class YoloAnnotator:
    def __init__(self, image_folder, output_folder="labels", class_id=0):
        self.image_folder = image_folder
        self.output_folder = Path(output_folder)
        self.class_id = class_id

        # Create the output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.current_image = None
        self.original_image = None
        self.image_name = ""
        self.image_list = []
        self.current_index = 0

        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.confirmed_boxes = []

        self.load_image_list()

        # Colors
        self.box_color = (0, 255, 0)  # Green for confirmed boxes
        self.temp_color = (0, 255, 255)  # Yellow for current drawing

    def load_image_list(self):
        """Load list of image files from the specified folder."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp', '*.webp']
        self.image_list = []
        for ext in extensions:
            self.image_list.extend(glob.glob(os.path.join(self.image_folder, ext.lower())))
            self.image_list.extend(glob.glob(os.path.join(self.image_folder, ext.upper())))
        
        # Sort the list using the natural_sort_key to handle numbers correctly
        self.image_list = sorted(list(set(self.image_list)), key=natural_sort_key)
        
        print(f"Found {len(self.image_list)} images in '{self.image_folder}'")
        if not self.image_list:
            raise ValueError(f"No images found in {self.image_folder}")

    def yolo_to_pixel(self, yolo_box, img_width, img_height):
        """Convert YOLO normalized coordinates to pixel coordinates (x1, y1, x2, y2)."""
        _, x_center_norm, y_center_norm, width_norm, height_norm = yolo_box

        box_width = width_norm * img_width
        box_height = height_norm * img_height
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height

        x1 = int(x_center - (box_width / 2))
        y1 = int(y_center - (box_height / 2))
        x2 = int(x_center + (box_width / 2))
        y2 = int(y_center + (box_height / 2))
        
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    def load_current_image(self):
        """Load the current image and its existing YOLO annotations if they exist."""
        if self.current_index >= len(self.image_list):
            return False

        image_path = self.image_list[self.current_index]
        self.image_name = os.path.basename(image_path)
        
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Could not load image: {image_path}")
            return False

        img_height, img_width, _ = self.original_image.shape
        self.current_image = self.original_image.copy()

        # Load existing YOLO boxes for this image
        self.confirmed_boxes = []
        label_filename = Path(self.image_name).stem + '.txt'
        label_path = self.output_folder / label_filename
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # Parse YOLO format: class_id, x_center, y_center, width, height
                        yolo_data = [float(p) for p in parts]
                        pixel_box = self.yolo_to_pixel(yolo_data, img_width, img_height)
                        self.confirmed_boxes.append(pixel_box)
            print(f"Loaded {len(self.confirmed_boxes)} boxes for {self.image_name}")

        self.draw_boxes()
        return True

    def draw_boxes(self):
        """Draw all confirmed and temporary boxes on the current image."""
        self.current_image = self.original_image.copy()

        # Draw confirmed boxes
        for i, box in enumerate(self.confirmed_boxes):
            cv2.rectangle(self.current_image, (box['x1'], box['y1']), (box['x2'], box['y2']), self.box_color, 2)
            # Add box number for easy identification
            cv2.putText(self.current_image, str(i + 1), (box['x1'], box['y1'] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2)

        # Draw the current box being drawn
        if self.drawing and self.start_point != (-1, -1):
            cv2.rectangle(self.current_image, self.start_point, self.end_point, self.temp_color, 2)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing and deleting bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.draw_boxes()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                # Ensure the box has a minimum size
                if abs(self.end_point[0] - self.start_point[0]) > 5 and \
                   abs(self.end_point[1] - self.start_point[1]) > 5:
                    
                    x1 = min(self.start_point[0], self.end_point[0])
                    y1 = min(self.start_point[1], self.end_point[1])
                    x2 = max(self.start_point[0], self.end_point[0])
                    y2 = max(self.start_point[1], self.end_point[1])
                    self.confirmed_boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                
                self.draw_boxes()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove the box if the right-click is inside it
            for i, box in reversed(list(enumerate(self.confirmed_boxes))):
                if box['x1'] <= x <= box['x2'] and box['y1'] <= y <= box['y2']:
                    self.confirmed_boxes.pop(i)
                    self.draw_boxes()
                    break

    def save_annotations_for_current_image(self):
        """Save annotations for the current image to a .txt file in YOLO format."""
        if not self.original_image.any():
            return

        img_height, img_width, _ = self.original_image.shape
        label_filename = Path(self.image_name).stem + '.txt'
        label_path = self.output_folder / label_filename

        with open(label_path, 'w') as f:
            for box in self.confirmed_boxes:
                # Get pixel coordinates
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                # Calculate box center and dimensions
                box_width = x2 - x1
                box_height = y2 - y1
                x_center = x1 + (box_width / 2)
                y_center = y1 + (box_height / 2)

                # Normalize coordinates
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = box_width / img_width
                height_norm = box_height / img_height

                # Write to file in YOLO format
                f.write(f"{self.class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        print(f"Saved {len(self.confirmed_boxes)} boxes to {label_path}")

    def show_instructions(self):
        """Display controls and status in the console."""
        instructions = f"""
        --- YOLO ANNOTATION TOOL ---
        Image {self.current_index + 1}/{len(self.image_list)}: {self.image_name}
        Boxes: {len(self.confirmed_boxes)}

        CONTROLS:
        - Left-click and drag: Draw a box
        - Right-click inside a box: Delete it
        - 'n': Save and go to NEXT image
        - 'p': Save and go to PREVIOUS image
        - 's': SAVE annotations for current image
        - 'r': RESET all boxes on current image
        - 'h': Show this help message
        - 'q': QUIT (will also save current work)
        ----------------------------
        """
        print(instructions)

    def run(self):
        """Main annotation loop."""
        if not self.load_current_image():
            print("Could not load the first image. Exiting.")
            return

        cv2.namedWindow('YOLO Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('YOLO Annotator', self.mouse_callback)
        self.show_instructions()

        while True:
            # Display status on the image window
            status_img = self.current_image.copy()
            status_text = f"Img: {self.current_index + 1}/{len(self.image_list)} | Boxes: {len(self.confirmed_boxes)} | Press 'h' for help"
            cv2.putText(status_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(status_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('YOLO Annotator', status_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.save_annotations_for_current_image()
                break
            
            elif key == ord('s'):
                self.save_annotations_for_current_image()
            
            elif key == ord('n'):
                self.save_annotations_for_current_image()
                self.current_index += 1
                if self.current_index >= len(self.image_list):
                    print("This is the last image.")
                    self.current_index = len(self.image_list) - 1
                self.load_current_image()
            
            elif key == ord('p'):
                self.save_annotations_for_current_image()
                self.current_index -= 1
                if self.current_index < 0:
                    print("This is the first image.")
                    self.current_index = 0
                self.load_current_image()

            elif key == ord('r'):
                self.confirmed_boxes = []
                self.draw_boxes()
                print(f"Cleared all boxes for {self.image_name}")

            elif key == ord('h'):
                self.show_instructions()

        cv2.destroyAllWindows()

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # 1. Folder containing your images
    IMAGE_FOLDER = "images"  # <<< CHANGE THIS

    # 2. Folder where YOLO .txt files will be saved
    LABEL_FOLDER = "labels"  # <<< CHANGE THIS

    # 3. The class ID for the objects you are annotating (e.g., 0 for 'tree')
    CLASS_ID = 0

    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' not found.")
        print("Please create it and place your images inside.")
    else:
        annotator = YoloAnnotator(IMAGE_FOLDER, LABEL_FOLDER, CLASS_ID)
        annotator.run()
        
        print("\nAnnotation complete!")
        print(f"Label files are saved in the '{LABEL_FOLDER}' directory.")