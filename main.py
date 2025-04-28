import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class CaptionGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Narrator")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f2f6")
        self.root.minsize(900, 600)
        
        # Initialize variables
        self.image_path = None
        self.caption_model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.image_display = None
        
        # Set default paths
        self.model_path = "models/model.keras"
        self.tokenizer_path = "models/tokenizer.pkl"
        self.feature_extractor_path = "models/feature_extractor.keras"
        
        # Create UI elements
        self.create_ui()
        
        # Load models automatically
        self.load_models_automatic()
        
    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and description
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=10)
        
        title_label = ttk.Label(title_frame, text="Visual Narrator", 
                               font=("Helvetica", 24, "bold"))
        title_label.pack()
        
        desc_label = ttk.Label(title_frame, 
                              text="Upload an image to generate description",
                              font=("Helvetica", 12))
        desc_label.pack(pady=5)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Controls panel - simplified
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Upload button
        upload_btn = ttk.Button(controls_frame, text="Select Image", command=self.upload_image)
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Generate caption button
        self.generate_btn = ttk.Button(controls_frame, text="Generate Description", 
                                     command=self.generate_caption, state=tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.status_label = ttk.Label(controls_frame, text="Loading models...", font=("Helvetica", 11))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", 
                                          mode="indeterminate", length=150)
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.progress_bar.start()
        
        # Content area with split view
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create a PanedWindow to allow resizing between image and caption
        paned_window = ttk.PanedWindow(content_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left pane - Image preview
        image_section = ttk.Frame(paned_window)
        paned_window.add(image_section, weight=2)  # Image gets more space
        
        self.display_frame = ttk.LabelFrame(image_section, text="Image Preview")
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial message
        self.placeholder_label = ttk.Label(self.display_frame, 
                                        text="Upload an image to see preview",
                                        font=("Helvetica", 12))
        self.placeholder_label.pack(expand=True)
        
        # Caption overlay frame - Added below image preview
        self.caption_overlay_frame = ttk.LabelFrame(image_section, text="Image Description", height=100)
        self.caption_overlay_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Add a label for the caption
        self.overlay_caption_label = ttk.Label(self.caption_overlay_frame, 
                                            text="No caption generated yet",
                                            font=("Helvetica", 12),
                                            wraplength=400,
                                            justify=tk.CENTER)
        self.overlay_caption_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right pane - Caption panel
        caption_panel = ttk.LabelFrame(paned_window, text="Generated Description")
        paned_window.add(caption_panel, weight=1)  # Caption panel gets less space
        
        # Caption result with scrollable text widget
        caption_frame = ttk.Frame(caption_panel)
        caption_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Use a Text widget for caption to support formatting and scrolling
        self.caption_text = tk.Text(caption_frame, wrap=tk.WORD, font=("Helvetica", 12),
                                  bg="#ffffff", relief=tk.FLAT, padx=10, pady=10,
                                  height=10, width=30)
        self.caption_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(caption_frame, orient="vertical", command=self.caption_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.caption_text.config(yscrollcommand=scrollbar.set)
        
        # Set initial text and disable editing
        self.caption_text.insert(tk.END, "No description generated yet.\n\nSelect an image and click 'Generate Description' to create a description.")
        self.caption_text.config(state=tk.DISABLED)
        
        # Configure styles
        self.configure_styles()
    
    def configure_styles(self):
        # Configure ttk styles for a modern look
        style = ttk.Style()
        style.configure("TFrame", background="#f0f2f6")
        style.configure("TLabelframe", background="#f0f2f6")
        style.configure("TLabelframe.Label", font=("Helvetica", 12, "bold"))
        style.configure("TButton", padding=8, relief="flat", background="#4a86e8", font=("Helvetica", 11))
    
    def load_models_automatic(self):
        # Start loading in a separate thread to prevent UI freeze
        thread = threading.Thread(target=self._load_models_thread)
        thread.daemon = True
        thread.start()
    
    def _load_models_thread(self):
        try:
            # Load models
            self.caption_model = load_model(self.model_path)
            self.feature_extractor = load_model(self.feature_extractor_path)
            
            with open(self.tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            
            # Update UI on the main thread
            self.root.after(0, self._models_loaded_success)
            
        except Exception as e:
            # Handle errors
            self.root.after(0, lambda: self.update_status(f"Error loading models: {str(e)}", error=True))
    
    def _models_loaded_success(self):
        self.status_label.config(text="Models loaded successfully")
        self.progress_bar.stop()
        self.progress_bar.pack_forget()  # Hide the progress bar
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.image_path = file_path
            
            # Enable generate button 
            self.generate_btn.config(state=tk.NORMAL)
            
            # Display the image
            self.display_image(file_path)
            self.status_label.config(text=f"Image selected: {os.path.basename(file_path)}")
            
            # Reset caption overlay
            self.overlay_caption_label.config(text="No description generated yet")
    
    def display_image(self, image_path):
        # Clear previous content
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        try:
            # Open and resize the image
            img = Image.open(image_path)
            img = self.resize_image(img, (400, 350))  # Reduced height to make room for caption
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Create a label to display the image
            self.image_display = ttk.Label(self.display_frame, image=photo)
            self.image_display.image = photo  # Keep a reference
            self.image_display.pack(padx=10, pady=10, expand=True)
            
        except Exception as e:
            error_label = ttk.Label(self.display_frame, 
                                  text=f"Error loading image: {str(e)}",
                                  foreground="red")
            error_label.pack(expand=True)
    
    def resize_image(self, img, size):
        # Resize image while maintaining aspect ratio
        width, height = img.size
        ratio = min(size[0]/width, size[1]/height)
        new_size = (int(width * ratio), int(height * ratio))
        return img.resize(new_size, Image.LANCZOS)
    
    def generate_caption(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        if not self.caption_model or not self.feature_extractor or not self.tokenizer:
            messagebox.showerror("Error", "Models are still loading. Please wait.")
            return
        
        # Start the progress bar
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.progress_bar.start()
        self.generate_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Generating Description...")
        
        # Update caption area
        self.caption_text.config(state=tk.NORMAL)
        self.caption_text.delete(1.0, tk.END)
        self.caption_text.insert(tk.END, "Processing...")
        self.caption_text.config(state=tk.DISABLED)
        
        # Update overlay caption
        self.overlay_caption_label.config(text="Processing...")
        
        # Generate caption in a separate thread
        thread = threading.Thread(target=self._generate_caption_thread)
        thread.daemon = True
        thread.start()
    
    def _generate_caption_thread(self):
        try:
            # Load and preprocess the image
            img = load_img(self.image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Extract features
            image_features = self.feature_extractor.predict(img_array, verbose=0)
            
            # Generate caption
            in_text = "startseq"
            max_length = 34
            
            for i in range(max_length):
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=max_length)
                yhat = self.caption_model.predict([image_features, sequence], verbose=0)
                yhat_index = np.argmax(yhat)
                word = self.tokenizer.index_word.get(yhat_index, None)
                if word is None:
                    break
                in_text += " " + word
                if word == "endseq":
                    break
            
            final_caption = in_text.replace("startseq", "").replace("endseq", "").strip()
            
            # Capitalize first letter and add a period if needed
            if final_caption:
                final_caption = final_caption[0].upper() + final_caption[1:]
                if not final_caption.endswith(('.', '!', '?')):
                    final_caption += '.'
            
            # Update UI on the main thread
            self.root.after(0, lambda: self._caption_generated_success(final_caption))
            
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Error generating description: {str(e)}", error=True))
    
    def _caption_generated_success(self, caption):
        # Update UI with the generated caption
        self.caption_text.config(state=tk.NORMAL)
        self.caption_text.delete(1.0, tk.END)
        
        # Add the main caption with formatting
        self.caption_text.insert(tk.END, "GENERATED Description:\n\n", "heading")
        self.caption_text.insert(tk.END, caption + "\n\n", "description")
        
        # Add some analysis or details
        self.caption_text.insert(tk.END, "Image Analysis:\n", "subheading")
        self.caption_text.insert(tk.END, "• Description successfully generated\n")
        self.caption_text.insert(tk.END, f"• Caption length: {len(caption.split())} words\n")
        
        # Add timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.caption_text.insert(tk.END, f"\nGenerated on: {timestamp}", "timestamp")
        
        # Configure tags for styling
        self.caption_text.tag_configure("heading", font=("Helvetica", 14, "bold"))
        self.caption_text.tag_configure("caption", font=("Helvetica", 12, "bold"), foreground="#1E3A8A")
        self.caption_text.tag_configure("subheading", font=("Helvetica", 12, "bold"))
        self.caption_text.tag_configure("timestamp", font=("Helvetica", 10), foreground="gray")
        
        self.caption_text.config(state=tk.DISABLED)
        
        # Update caption overlay - This is key to displaying caption below the image
        self.overlay_caption_label.config(text=caption)
        
        # Update status and UI
        self.status_label.config(text="Description generated successfully")
        self.progress_bar.stop()
        self.progress_bar.pack_forget()  # Hide progress bar when done
        self.generate_btn.config(state=tk.NORMAL)
    
    def update_status(self, message, error=False):
        if error:
            messagebox.showerror("Error", message)
            self.status_label.config(text="Error loading models")
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            
            # Update caption area
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, f"Error: {message}")
            self.caption_text.config(state=tk.DISABLED)
            
            # Update overlay caption
            self.overlay_caption_label.config(text=f"Error: {message}")
        else:
            self.status_label.config(text=message)


def main():
    root = tk.Tk()
    app = CaptionGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()