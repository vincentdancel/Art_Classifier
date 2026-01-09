# train_model_debug.py
# Debug version to see what's happening

import os
import json
import tensorflow as tf
from keras import layers, models, optimizers, callbacks
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = "dataset"


class ArtClassifierTrainer:
    def __init__(self):
        self.class_names = []
        self.num_classes = 0
        self.model = None

    def prepare_data(self):
        train_dir = os.path.join(DATA_DIR, "train")
        
        print("\n" + "="*60)
        print("🔍 DEBUGGING DATASET STRUCTURE")
        print("="*60)
        
        print(f"\n📂 Train directory: {train_dir}")
        print(f"   Absolute path: {os.path.abspath(train_dir)}")
        
        # List everything in train directory
        all_items = os.listdir(train_dir)
        print(f"\n📋 All items in train/: {all_items}")
        
        # Check each folder
        print("\n🔍 Checking each folder:")
        for item in sorted(all_items):
            item_path = os.path.join(train_dir, item)
            if os.path.isdir(item_path):
                images = [f for f in os.listdir(item_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                print(f"   ✅ {item}: {len(images)} images")
                if len(images) > 0:
                    print(f"      Sample: {images[0]}")
            else:
                print(f"   ⚠️  {item} is a FILE (not a folder!)")
        
        # Get class names manually
        self.class_names = sorted([item for item in os.listdir(train_dir) 
                                  if os.path.isdir(os.path.join(train_dir, item))])
        self.num_classes = len(self.class_names)
        
        print(f"\n✅ Found {self.num_classes} classes: {self.class_names}")
        print("="*60 + "\n")

        # Save class indices
        class_indices = {name: i for i, name in enumerate(self.class_names)}
        with open("class_indices.json", "w", encoding="utf-8") as f:
            json.dump(class_indices, f, indent=4)
        
        print("📊 Creating data generators...")

        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            rotation_range=15,
            zoom_range=0.15,
            horizontal_flip=True
        )

        print("\n🔄 Creating training generator...")
        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training"
        )
        
        print(f"\n✅ Training generator created!")
        print(f"   Classes found by generator: {list(train_gen.class_indices.keys())}")
        print(f"   Total training images: {train_gen.samples}")
        print(f"   Class indices: {train_gen.class_indices}")

        print("\n🔄 Creating validation generator...")
        val_gen = datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation"
        )
        
        print(f"\n✅ Validation generator created!")
        print(f"   Classes found by generator: {list(val_gen.class_indices.keys())}")
        print(f"   Total validation images: {val_gen.samples}")
        
        # Check if classes match
        if len(train_gen.class_indices) != self.num_classes:
            print("\n⚠️  WARNING: Mismatch detected!")
            print(f"   Expected classes: {self.num_classes} ({self.class_names})")
            print(f"   Generator found: {len(train_gen.class_indices)} ({list(train_gen.class_indices.keys())})")
            print("\n💡 Possible reasons:")
            print("   1. Some folders are empty")
            print("   2. Some folders have no valid images")
            print("   3. Some folders don't have enough images for validation split")
            print("   4. Image files might be corrupted")

        return train_gen, val_gen

    def build_model(self):
        print("\n🏗️  Building model...")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights="imagenet"
        )

        base_model.trainable = False

        inputs = layers.Input(shape=(*IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)

        self.model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("\n📊 Model Summary:")
        self.model.summary()

    def train(self, train_gen, val_gen):
        print("\n🚀 Starting training...")
        os.makedirs("models", exist_ok=True)

        checkpoint = callbacks.ModelCheckpoint(
            "models/art_classifier_best",
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        )

        early_stop = callbacks.EarlyStopping(
            patience=7,
            restore_best_weights=True
        )

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stop]
        )

        self.model.save("models/art_classifier_final", save_format="tf")

        with open("labels.json", "w", encoding="utf-8") as f:
            json.dump({
                "artists": self.class_names,
                "total_classes": self.num_classes
            }, f, indent=4)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"📊 Classes trained: {self.class_names}")
        print(f"📁 Model saved to: models/art_classifier_best")


def main():
    trainer = ArtClassifierTrainer()
    train_gen, val_gen = trainer.prepare_data()
    
    # Stop here if there's a problem
    if len(train_gen.class_indices) <= 1:
        print("\n❌ ERROR: Only 1 class detected!")
        print("   Cannot train a classifier with just one class.")
        print("\n📋 Please check:")
        print("   1. All artist folders have images")
        print("   2. Image files are valid (.jpg, .png, etc)")
        print("   3. Each folder has at least 5 images")
        return
    
    trainer.build_model()
    trainer.train(train_gen, val_gen)


if __name__ == "__main__":
    main()