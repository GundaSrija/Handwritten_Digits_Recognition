import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageTk
import os
import threading
from queue import Queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DigitRecognitionApp:
    def __init__(self):
        self.model_path = 'mnist_model.h5'
        self.root = tk.Tk()
        self.root.title("Multiple Digit Recognition")
        self.digit_count = tk.IntVar(value=1)
        
        # Updated modern dark theme colors
        self.colors = {
            'background': '#1E1E1E',      # Dark background
            'secondary_bg': '#2D2D2D',    # Slightly lighter background
            'primary': '#00A9F4',         # Bright blue
            'secondary': '#FF4081',       # Pink
            'success': '#00E676',         # Bright green
            'warning': '#FFD740',         # Amber
            'error': '#FF5252',           # Red
            'text': '#FFFFFF',            # White text
            'text_secondary': '#B0B0B0',  # Grey text
            'accent': '#7C4DFF',          # Purple accent
        }
        
        # Configure window with dark theme
        self.root.configure(bg=self.colors['background'])
        
        # Configure window
        window_width = 600  # Increased width
        window_height = 500  # Increased height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.resizable(True, True)
        
        self.setup_main_menu()
        self.create_status_bar()

        # Add these new instance variables
        self.prediction_queue = Queue()
        self.is_predicting = False
        self.drawing_buffer = []
        
        # Add double buffering variable
        self.draw_update_pending = False
        self.draw_update_delay = 10  # milliseconds
        self.line_width = 15  # Reduced from 20 for better performance

    def setup_main_menu(self):
        # Main container with dark theme
        main_container = tk.Frame(self.root, bg=self.colors['background'], pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Simple title without animation
        title_label = tk.Label(main_container, 
                            text="Digit Recognition AI",
                            font=('Helvetica', 32, 'bold'),
                            bg=self.colors['background'],
                            fg=self.colors['primary'])
        title_label.pack(pady=(0, 30))
        
        # Button container
        button_frame = tk.Frame(main_container, bg=self.colors['background'])
        button_frame.pack(fill=tk.BOTH, expand=True, padx=40)
        
        buttons = [
            ("Train Model", self.train_model, "Train a new model on MNIST dataset"),
            ("Draw & Predict", self.open_drawing_interface, "Open drawing canvas to predict digits"),
            ("Upload & Predict", self.upload_image, "Upload an image file for prediction"),
            ("View MNIST Samples", self.view_mnist_samples, "View all MNIST dataset samples"),
            ("View Model Details", self.view_model_details, "View model architecture and weights"),
            ("Quit", self.root.quit, "Exit the application")
        ]
        
        for text, command, tooltip in buttons:
            btn_container = tk.Frame(button_frame, bg=self.colors['background'])
            btn_container.pack(pady=8, fill=tk.X)
            
            btn = tk.Button(btn_container, 
                           text=text,
                           command=command,
                           font=('Helvetica', 14),
                           bg=self.colors['secondary_bg'],
                           fg=self.colors['text'],
                           activebackground=self.colors['primary'],
                           activeforeground=self.colors['text'],
                           relief=tk.FLAT,
                           pady=12,
                           width=30)
            btn.pack(fill=tk.X)
            
            # Simple tooltip without animations
            self.create_modern_tooltip(btn, tooltip)
        
        # Modern configuration frame
        config_frame = tk.LabelFrame(main_container, 
                                   text=" Configuration ", 
                                   font=('Helvetica', 12, 'bold'),
                                   bg=self.colors['secondary_bg'],
                                   fg=self.colors['text'],  # Changed from primary to text for better visibility
                                   pady=15,
                                   padx=10)
        config_frame.pack(fill=tk.X, pady=(20, 0), padx=40)
        
        # Digit selector with improved modern styling
        tk.Label(config_frame, 
                text="Number of digits:",
                font=('Helvetica', 12, 'bold'),  # Made font bold and larger
                bg=self.colors['secondary_bg'],
                fg=self.colors['text']).pack(side=tk.LEFT, padx=(10, 0))
        
        digit_spin = ttk.Spinbox(
            config_frame,
            from_=1,
            to=10,  # Changed from 5 to 10
            width=5,
            textvariable=self.digit_count,
            font=('Helvetica', 12, 'bold')  # Made font bold and larger
        )
        digit_spin.pack(side=tk.LEFT, padx=10)
        
        # Updated Style configuration for better visibility
        style = ttk.Style()
        style.configure('TSpinbox',
                       fieldbackground=self.colors['primary'],  # Changed to primary color
                       foreground=self.colors['text'],
                       arrowcolor=self.colors['text'])

    def create_modern_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Label(widget,
                          text=text,
                          bg=self.colors['accent'],
                          fg=self.colors['text'],
                          relief='solid',
                          borderwidth=0,
                          font=('Helvetica', 10))
            tooltip.place(x=widget.winfo_width() + 5, y=0)
            widget.tooltip = tooltip
            
        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')
            
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=self.colors['primary'],
            fg='white',
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=10,
            pady=5,
            font=('Helvetica', 10)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_and_preprocess_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        
        return (x_train, y_train), (x_test, y_test)

    def create_model(self):
        # Enhanced model architecture
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        return model

    def train_model(self):
        train_window = tk.Toplevel(self.root)
        train_window.title("Training Progress")
        progress_label = ttk.Label(train_window, text="Training in progress...")
        progress_label.pack(pady=20)

        def training_thread():
            try:
                (x_train, y_train), (x_test, y_test) = self.load_and_preprocess_data()
                model = self.create_model()
                
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

                datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=15,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    zoom_range=0.15,
                    shear_range=0.15
                )

                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=5, restore_best_weights=True)
                
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy', factor=0.2, patience=3)

                history = model.fit(
                    datagen.flow(x_train, y_train, batch_size=32),
                    epochs=50,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, reduce_lr]
                )

                model.save(self.model_path)
                
                test_loss, test_accuracy = model.evaluate(x_test, y_test)
                train_window.after(0, lambda: messagebox.showinfo("Training Complete", 
                                  f"Model trained successfully!\nTest accuracy: {test_accuracy*100:.2f}%"))
                train_window.after(0, train_window.destroy)
            except Exception as e:
                train_window.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
                train_window.after(0, train_window.destroy)

        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()

    def open_drawing_interface(self):
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Warning", "Please train the model first!")
            return

        draw_window = tk.Toplevel(self.root)
        draw_window.title("Draw Digits")
        draw_window.configure(bg=self.colors['background'])
        
        # Enable double buffering for the canvas
        canvas_frame = tk.Frame(draw_window, 
                              bg=self.colors['primary'],
                              padx=2, pady=2)
        canvas_frame.pack(pady=20)
        
        canvas_width = 280 * self.digit_count.get()
        self.canvas = tk.Canvas(canvas_frame, 
                              width=canvas_width,
                              height=280,
                              bg='black',
                              cursor='pencil')
        self.canvas.pack()
        
        # Add guide lines with custom style
        for i in range(1, self.digit_count.get()):
            x = i * 280
            self.canvas.create_line(x, 0, x, 280,
                                  fill=self.colors['secondary'],
                                  dash=(4, 4),
                                  width=2)

        # Styled buttons
        button_frame = tk.Frame(draw_window, bg=self.colors['background'])
        button_frame.pack(pady=20)
        
        clear_btn = tk.Button(button_frame,
                            text="Clear",
                            command=self.clear_canvas,
                            bg=self.colors['error'],
                            fg='white',
                            font=('Helvetica', 12),
                            padx=20, pady=5,
                            cursor='hand2')
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        predict_btn = tk.Button(button_frame,
                              text="Predict",
                              command=self.predict,
                              bg=self.colors['success'],
                              fg='white',
                              font=('Helvetica', 12),
                              padx=20, pady=5,
                              cursor='hand2')
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Result label
        self.result_label = tk.Label(draw_window,
                                   text="Draw digits and click Predict",
                                   font=('Helvetica', 14),
                                   bg=self.colors['background'],
                                   fg=self.colors['primary'])
        self.result_label.pack(pady=20)
        
        # Drawing variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                 fill='white', width=20)
            self.last_x = x
            self.last_y = y

    def stop_drawing(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="Draw digits and click Predict")

    def save_canvas(self):
        # Create in-memory image
        image = Image.new("L", (self.canvas.winfo_width(), 280), "black")
        draw = ImageDraw.Draw(image)
        
        self.canvas.update()
        bbox = self.canvas.bbox("all")
        if bbox:
            for item in self.canvas.find_all():
                coords = self.canvas.coords(item)
                draw.line(coords, fill="white", width=20)
        
        temp_path = "temp_digit.png"
        image.save(temp_path)
        return temp_path

    def segment_digits(self, image):
        width = image.width
        digit_width = width // self.digit_count.get()
        digits = []
        
        for i in range(self.digit_count.get()):
            left = i * digit_width
            right = (i + 1) * digit_width
            digit = image.crop((left, 0, right, image.height))
            digits.append(digit)
            
        return digits

    def process_image_for_prediction(self, image_path):
        """Simple and robust image processing for single digit prediction"""
        try:
            # Open image with PIL
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Resize to comfortable size while maintaining aspect ratio
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Apply thresholding to create binary image
            # If image is inverted (white digit on black), invert it
            if np.mean(img_array) > 127:
                # Dark digit on light background
                _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                # Light digit on dark background
                _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find digit contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours, just use the whole image
            if not contours:
                processed = cv2.resize(binary, (28, 28))
                return processed
            
            # Find the largest contour (assumed to be the digit)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding (10% of the larger dimension)
            padding = int(max(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(binary.shape[1], x + w + padding)
            y2 = min(binary.shape[0], y + h + padding)
            
            # Extract digit with padding
            digit = binary[y1:y2, x1:x2]
            
            # Make square by adding padding
            height, width = digit.shape
            if height > width:
                # Add padding to width
                diff = height - width
                pad_left = diff // 2
                pad_right = diff - pad_left
                digit = cv2.copyMakeBorder(digit, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            elif width > height:
                # Add padding to height
                diff = width - height
                pad_top = diff // 2
                pad_bottom = diff - pad_top
                digit = cv2.copyMakeBorder(digit, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
            
            # Resize to MNIST standard (28x28)
            processed = cv2.resize(digit, (28, 28))
            
            # Save for debugging
            cv2.imwrite("processed_digit.png", processed)
            
            return processed
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

    def upload_image(self):
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Warning", "Please train the model first!")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not file_path:
            return

        processing_window = tk.Toplevel(self.root)
        processing_window.title("Processing Image")
        processing_window.geometry("300x100")
        
        progress_label = ttk.Label(processing_window, text="Processing image, please wait...")
        progress_label.pack(pady=20)
        
        def processing_thread():
            try:
                processed_digit = self.process_image_for_prediction(file_path)
                model = tf.keras.models.load_model(self.model_path)
                
                img_array = processed_digit.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                img_array = np.expand_dims(img_array, axis=-1)
                
                prediction = model.predict(img_array, verbose=0)
                digit = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                
                display_digit = Image.fromarray(processed_digit)
                
                def update_ui():
                    processing_window.destroy()
                    result = f"Predicted digit: {digit}\nConfidence: {confidence:.2f}%"
                    messagebox.showinfo("Prediction Result", result)
                    self.show_processed_digits([display_digit], [digit])
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                self.root.after(0, processing_window.destroy)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to process image: {str(e)}"))

        thread = threading.Thread(target=processing_thread, daemon=True)
        thread.start()

    def show_processed_digits(self, digits, predictions):
        """Show the processed digits for verification"""
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Processed Digits")
        
        # Calculate window size based on number of digits
        window_width = min(800, len(digits) * 100 + 50)
        preview_window.geometry(f"{window_width}x200")
        
        # Create a frame for the digits
        digits_frame = ttk.Frame(preview_window)
        digits_frame.pack(pady=20)
        
        # Display each processed digit with its prediction
        for i, (digit_img, pred) in enumerate(zip(digits, predictions)):
            # Create a frame for this digit
            digit_frame = ttk.Frame(digits_frame)
            digit_frame.pack(side=tk.LEFT, padx=10)
            
            # Resize for display
            display_img = digit_img.resize((80, 80), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo_img = ImageTk.PhotoImage(display_img)
            
            # Create label for image
            img_label = ttk.Label(digit_frame, image=photo_img)
            img_label.image = photo_img  # Keep a reference
            img_label.pack()
            
            # Create label for prediction
            pred_label = ttk.Label(digit_frame, text=f"Digit: {pred}")
            pred_label.pack()

    def predict(self):
        if self.is_predicting:
            return
        
        self.is_predicting = True
        self.result_label.config(text="Predicting...")
        
        def enable_ui():
            self.canvas.config(state='normal')
            self.is_predicting = False

        def prediction_thread():
            try:
                temp_path = self.save_canvas()
                full_image = Image.open(temp_path)
                digits = self.segment_digits(full_image)
                
                # Load model
                if not hasattr(self, '_model'):
                    self._model = tf.keras.models.load_model(self.model_path)
                
                predictions = []
                confidences = []
                
                for digit_img in digits:
                    # Preprocess individual digit
                    digit_img = digit_img.resize((28, 28), Image.Resampling.LANCZOS)
                    img_array = np.array(digit_img)
                    img_array = img_array.astype('float32') / 255.0
                    img_array = np.expand_dims(img_array, axis=-1)
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    prediction = self._model.predict(img_array, verbose=0)
                    digit = np.argmax(prediction[0])
                    confidence = prediction[0][digit] * 100
                    
                    predictions.append(digit)
                    confidences.append(confidence)
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Update result
                result_text = "Predicted number: " + "".join(map(str, predictions))
                result_text += f"\nConfidence: {np.mean(confidences):.2f}%"
                self.root.after(0, lambda: self.result_label.config(text=result_text))
                
            except Exception as e:
                self.root.after(0, lambda: self.result_label.config(text=f"Error: {str(e)}"))
            finally:
                self.root.after(0, enable_ui)
        
        thread = threading.Thread(target=prediction_thread, daemon=True)
        thread.start()

    def view_mnist_samples(self):
        # Create a new window
        samples_window = tk.Toplevel(self.root)
        samples_window.title("MNIST Dataset Samples")
        
        def load_and_display():
            try:
                # Load MNIST data
                (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
                
                # Create figure with scrollable canvas
                fig = plt.figure(figsize=(20, 10))  # Adjust figure size
                rows = 10  # Number of rows
                cols = 10  # Number of columns
                
                for i in range(rows * cols):  # Show up to rows * cols samples
                    ax = plt.subplot(rows, cols, i + 1)
                    ax.imshow(x_train[i], cmap='gray')
                    ax.axis('off')
                
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                
                # Create canvas widget
                canvas = FigureCanvasTkAgg(fig, master=samples_window)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load MNIST samples: {str(e)}")
                samples_window.destroy()
        
        # Run in thread to prevent UI freezing
        thread = threading.Thread(target=load_and_display, daemon=True)
        thread.start()

    def view_model_details(self):
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Warning", "No model file found. Please train the model first!")
            return
            
        try:
            model = tf.keras.models.load_model(self.model_path)
            
            details_window = tk.Toplevel(self.root)
            details_window.title("Model Details")
            details_window.geometry("800x600")
            details_window.configure(bg=self.colors['background'])
            
            # Create tabbed interface
            notebook = ttk.Notebook(details_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Architecture Tab
            arch_frame = ttk.Frame(notebook)
            notebook.add(arch_frame, text='Architecture')
            
            arch_text = tk.Text(arch_frame,
                              bg=self.colors['secondary_bg'],
                              fg=self.colors['text'],
                              font=('Courier', 10))
            arch_text.pack(fill=tk.BOTH, expand=True)
            
            # Model Summary
            arch_text.insert(tk.END, "MODEL ARCHITECTURE SUMMARY\n")
            arch_text.insert(tk.END, "========================\n\n")
            arch_text.insert(tk.END, "This CNN model consists of:\n")
            arch_text.insert(tk.END, "- Multiple Convolutional layers for feature extraction\n")
            arch_text.insert(tk.END, "- BatchNormalization for training stability\n")
            arch_text.insert(tk.END, "- MaxPooling layers for dimension reduction\n")
            arch_text.insert(tk.END, "- Dropout layers to prevent overfitting\n")
            arch_text.insert(tk.END, "- Dense layers for classification\n\n")
            
            # Detailed layer information
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            arch_text.insert(tk.END, '\n'.join(stringlist))
            arch_text.configure(state='disabled')
            
            # Layer Details Tab
            layer_frame = ttk.Frame(notebook)
            notebook.add(layer_frame, text='Layer Details')
            
            layer_text = tk.Text(layer_frame,
                               bg=self.colors['secondary_bg'],
                               fg=self.colors['text'],
                               font=('Courier', 10))
            layer_text.pack(fill=tk.BOTH, expand=True)
            
            # Detailed information about each layer
            for i, layer in enumerate(model.layers):
                layer_text.insert(tk.END, f"\nLAYER {i}: {layer.name.upper()}\n")
                layer_text.insert(tk.END, "="*(len(f"LAYER {i}: {layer.name}")+1) + "\n")
                layer_text.insert(tk.END, f"Type: {layer.__class__.__name__}\n")
                layer_text.insert(tk.END, f"Input Shape: {layer.input_shape}\n")
                layer_text.insert(tk.END, f"Output Shape: {layer.output_shape}\n")
                layer_text.insert(tk.END, f"Trainable Parameters: {layer.count_params():,}\n")
                
                # Add layer-specific information
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_text.insert(tk.END, f"Filters: {layer.filters}\n")
                    layer_text.insert(tk.END, f"Kernel Size: {layer.kernel_size}\n")
                    layer_text.insert(tk.END, f"Activation: {layer.activation.__name__}\n")
                elif isinstance(layer, tf.keras.layers.Dense):
                    layer_text.insert(tk.END, f"Units: {layer.units}\n")
                    layer_text.insert(tk.END, f"Activation: {layer.activation.__name__}\n")
                elif isinstance(layer, tf.keras.layers.Dropout):
                    layer_text.insert(tk.END, f"Dropout Rate: {layer.rate}\n")
            
            layer_text.configure(state='disabled')
            
            # Statistics Tab
            stats_frame = ttk.Frame(notebook)
            notebook.add(stats_frame, text='Statistics')
            
            stats_text = tk.Text(stats_frame,
                               bg=self.colors['secondary_bg'],
                               fg=self.colors['text'],
                               font=('Courier', 10))
            stats_text.pack(fill=tk.BOTH, expand=True)
            
            # Model statistics
            total_params = model.count_params()
            trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
            non_trainable_params = total_params - trainable_params
            
            stats_text.insert(tk.END, "MODEL STATISTICS\n")
            stats_text.insert(tk.END, "===============\n\n")
            stats_text.insert(tk.END, f"Total Parameters: {total_params:,}\n")
            stats_text.insert(tk.END, f"Trainable Parameters: {trainable_params:,}\n")
            stats_text.insert(tk.END, f"Non-trainable Parameters: {non_trainable_params:,}\n")
            stats_text.insert(tk.END, f"Model File Size: {os.path.getsize(self.model_path) / (1024*1024):.2f} MB\n")
            stats_text.insert(tk.END, f"Number of Layers: {len(model.layers)}\n")
            stats_text.configure(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model details: {str(e)}")
            details_window.destroy()

    def run(self):
        self.root.mainloop()
        

if __name__ == "__main__":
    app = DigitRecognitionApp()
    app.run()