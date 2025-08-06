
import os
import time
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from skimage.feature import hog

# === TRANSFORMS ===
TARGET_SIZE = (224, 224)
img_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === IMAGE PREPROCESSING ===
def rotate_and_crop_centered_no_black(img, angle, crop_size=(224, 224), padding_ratio=1.5):
    original_size = max(img.size)
    oversize = int(original_size * padding_ratio)
    img = img.resize((oversize, oversize), resample=Image.BICUBIC)
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    w, h = rotated.size
    tw, th = crop_size
    left = (w - tw) // 2
    upper = (h - th) // 2
    right = left + tw
    lower = upper + th
    return rotated.crop((left, upper, right, lower))

def random_augmentation_clean(img, rotation_max=15, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
    angle = random.uniform(-rotation_max, rotation_max)
    img = rotate_and_crop_centered_no_black(img, angle, crop_size=(224, 224))
    flip_chance = random.random()
    if flip_chance < 0.25:
        pass
    elif flip_chance < 0.50:
        img = ImageOps.flip(img)
    elif flip_chance < 0.75:
        img = ImageOps.mirror(img)
    else:
        img = img.rotate(180)
    brightness_factor = random.uniform(*brightness_range)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    contrast_factor = random.uniform(*contrast_range)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img

def generate_augmented_images(img, n_augmented=4):
    img = img.resize((224, 224))
    versions = [img]
    for _ in range(n_augmented):
        versions.append(random_augmentation_clean(img))
    return versions

# === NOISE FUNCTIONS ===
def apply_salt_and_pepper_noise(img, amount=0.4):
    arr = np.array(img)
    output = arr.copy()
    h, w, c = output.shape
    num_salt = np.ceil(amount * h * w * 0.5).astype(int)
    num_pepper = np.ceil(amount * h * w * 0.5).astype(int)
    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    output[coords[0], coords[1]] = [255, 255, 255]
    coords = (np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper))
    output[coords[0], coords[1]] = [0, 0, 0]
    return Image.fromarray(output)

def apply_gaussian_noise(img, mean=0, sigma=200):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_speckle_noise(img, var=0.2):
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.randn(*arr.shape) * var
    noisy = arr + arr * noise
    noisy = np.clip(noisy, 0, 1) * 255
    return Image.fromarray(noisy.astype(np.uint8))

# === DATASET LOADER ===
def load_images_with_augmentation_and_noise(root_dir, brands_dict,
                                            salt_pepper_amount,
                                            gaussian_sigma,
                                            speckle_var):
    dataset = {
        "augmented": [],
        "salt_pepper": [],
        "gaussian": [],
        "speckle": []
    }
    for brand, count in brands_dict.items():
        brand_path = os.path.join(root_dir, brand)
        for i in range(1, count + 1):
            filename = f"{brand} ({i}).jpg"
            filepath = os.path.join(brand_path, filename)
            try:
                img = Image.open(filepath).convert("RGB")
                versions = generate_augmented_images(img, n_augmented=4)
                for version in versions:
                    version_np = np.array(version)
                    dataset["augmented"].append({"image": version_np, "label": brand})
                    dataset["salt_pepper"].append({
                        "image": np.array(apply_salt_and_pepper_noise(version, amount=salt_pepper_amount)),
                        "label": brand})
                    dataset["gaussian"].append({
                        "image": np.array(apply_gaussian_noise(version, sigma=gaussian_sigma)),
                        "label": brand})
                    dataset["speckle"].append({
                        "image": np.array(apply_speckle_noise(version, var=speckle_var)),
                        "label": brand})
            except Exception as e:
                print(f"[ERROR] Could not load {filepath}: {e}")
    return dataset


# === VISUALIZATION ===
def show_noise_comparison_grid(dataset_dict, brands):
    random.seed(time.time())
    fig, axs = plt.subplots(4, 8, figsize=(18, 9))
    versions = ["augmented", "salt_pepper", "gaussian", "speckle"]
    for col, brand in enumerate(brands):
        idx_list = [i for i, item in enumerate(dataset_dict["augmented"]) if item["label"] == brand]
        if not idx_list:
            continue
        idx = random.choice(idx_list)
        for row, version in enumerate(versions):
            axs[row, col].imshow(dataset_dict[version][idx]["image"])
            axs[row, col].axis("off")
            if row == 0:
                axs[row, col].set_title(brand)
    plt.tight_layout()
    plt.show()

# === METRICS ===

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

# === DATASET CLASS ===
class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# === TRAINING FUNCTIONS ===
def train_model(model, train_loader, val_loader, model_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    start_time_model = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.numpy())
        if epoch == EPOCHS - 1:
            elapsed_model_time = round(time.time() - start_time_model, 2)
            metrics = compute_metrics(all_labels, all_preds)
            print(f"[{model_name}] Final Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")
            print(f"[{model_name}] Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} \n(Time: {elapsed_model_time}s)")
        else:
            print(f"[{model_name}] Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")
    return model, metrics

def extract_features(images):
    feats = []
    for img in images:
        gray = np.mean(img, axis=2).astype(np.uint8)
        hist = np.histogram(gray, bins=32, range=(0, 255))[0]
        h = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        feats.append(np.hstack([hist, h]))
    return np.array(feats)

def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = compute_metrics(y_val, preds)
    
    print(f"[XGBoost] Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    return model, metrics

def train_knn(X_train, y_train, X_val, y_val):
    model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = compute_metrics(y_val, preds)
    
    print(f"[KNN] Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    return model, metrics

# === TRAIN ALL MODELS ===
def train_all_models(dataset_dict):
    label_encoder = LabelEncoder()
    dataset_results = {}
    for variant_name, data in dataset_dict.items():
        print(f"\n=== Training on: {variant_name} ===")
        print(f"\nInitializing dataset...")
        variant_start = time.time()
        images = [x['image'] for x in data]
        labels = [x['label'] for x in data]
        labels_encoded = label_encoder.fit_transform(labels)
        X_train, X_temp, y_train, y_temp = train_test_split(images, labels_encoded, test_size=0.3, stratify=labels_encoded, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        train_ds = ImageDataset(X_train, y_train, transform=img_transform)
        val_ds = ImageDataset(X_val, y_val, transform=img_transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        model_times = {}
        variant_metrics = {}

        start = time.time()
        X_train_feats = extract_features(X_train)
        X_val_feats = extract_features(X_val)
        
        print("\n====================================")
        print(">>> KNN")
        print("KNN training...")
        start_time_knn = time.time()
        _, metrics_knn = train_knn(X_train_feats, y_train, X_val_feats, y_val)
        model_times['knn_time'] = round(time.time() - start, 2)
        elapsed_knn_time = round(time.time() - start_time_knn, 2)
        variant_metrics['knn_metrics'] = metrics_knn
        print(f"(Time: {elapsed_knn_time}s)")

        start = time.time()
        
        print("\n====================================")
        print(">>> XGBoost")
        print("XGBoost training...")
        start_time_xgb = time.time()
        _, metrics_xgb = train_xgboost(X_train_feats, y_train, X_val_feats, y_val)
        model_times['xgb_time'] = round(time.time() - start, 2)
        elapsed_xgb_time = round(time.time() - start_time_xgb, 2)
        variant_metrics['xgb_metrics'] = metrics_xgb
        print(f"(Time: {elapsed_xgb_time}s)")

        start = time.time()
        
        print("\n====================================")
        print(">>> EfficientNet-B0")
        print("EfficientNet-B0 training...")
        model_cnn = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model_cnn.classifier[1] = nn.Linear(model_cnn.classifier[1].in_features, NUM_CLASSES)
        _, metrics_cnn = train_model(model_cnn, train_loader, val_loader, "EfficientNet-B0")
        model_times['cnn_time'] = round(time.time() - start, 2)
        variant_metrics['cnn_metrics'] = metrics_cnn

        start = time.time()
        
        print("\n====================================")
        print(">>> ViT-B16")
        print("ViT-B16 training...")
        model_vit = models.vit_b_16(weights='IMAGENET1K_V1')
        model_vit.heads.head = nn.Linear(model_vit.heads.head.in_features, NUM_CLASSES)
        _, metrics_vit = train_model(model_vit, train_loader, val_loader, "ViT-B16")
        model_times['vit_time'] = round(time.time() - start, 2)
        variant_metrics['vit_metrics'] = metrics_vit

        dataset_results[variant_name] = {
            'metrics': variant_metrics,
            'model_times_seconds': model_times,
            'total_time_seconds': round(time.time() - variant_start, 2)
        }
    return dataset_results

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("=== START ===")
    # === MODEL CONFIGURATION ===
    KNN_NEIGHBORS = 4
    BATCH_SIZE = 32
    EPOCHS = 5
    # === SYSTEM CONFIGURATION ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRINT_INTERVAL = 0
    NUM_CLASSES = 8
    # === NOISE PARAMETERS ===
    SALT_PEPPER_AMOUNT = 0.4
    GAUSSIAN_SIGMA = 200
    SPECKLE_VAR = 0.2
    # === DATASET CONFIGURATION ===
    ROOT_DIR = 'Master-s-thesis\\photos'
    BRANDS = {
        'Audi': 34,'Citroen': 23,'Mazda': 21,'Mercedes': 21,
        'Opel': 29,'Renault': 24,'Skoda': 21,'Toyota': 46
    }
    print('[INFO] Preparing dataset...')
    dataset = load_images_with_augmentation_and_noise(
        ROOT_DIR, BRANDS,
        salt_pepper_amount=SALT_PEPPER_AMOUNT,
        gaussian_sigma=GAUSSIAN_SIGMA,
        speckle_var=SPECKLE_VAR
    )
    print('[INFO] Preparing noise preview images...')
    show_noise_comparison_grid(dataset, list(BRANDS.keys()))
    results = train_all_models(dataset)
    print("\n=== FINAL RESULTS ===")
    for variant, data in results.items():
        print(f"\n[{variant.upper()}]")
        for model, metrics in data['metrics'].items():
            print(f"  {model}: {metrics}")
        print(f"  Times: {data['model_times_seconds']}")
        print(f"  Total time: {data['total_time_seconds']}s")
    print("=== DONE ===")
