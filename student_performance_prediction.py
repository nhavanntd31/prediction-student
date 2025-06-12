import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
import argparse
warnings.filterwarnings('ignore')

class StudentPerformanceDataProcessor:
    def __init__(self):
        self.grade_mapping = {
            'A+': 4.0, 'A': 4, 'B+': 3.5, 'B': 3, 'C+': 2.5, 'C': 2,
            'D+': 1.5, 'D': 1, 'F': 0, 'X': 0, 'W': 0
        }
        self.warning_mapping = {
            'Mức 0': 0, 'Mức 1': 1, 'Mức 2': 2, 'Mức 3': 3
        }
        self.scaler = StandardScaler()
        
    def clean_numeric_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            df[col] = df[col].replace([np.inf, -np.inf], mean_val)
        return df
        
    def preprocess_course_data(self, course_df: pd.DataFrame) -> pd.DataFrame:
        df = course_df.copy()
        
        df['Final Grade Numeric'] = df['Final Grade'].map(self.grade_mapping)
        
        numeric_features = ['Continuous Assessment Score', 'Exam Score', 'Credits', 
                          'Final Grade Numeric']
        
        df = self.clean_numeric_data(df, numeric_features)
        
        df['Course_Category'] = df['Course ID'].str[:2]
        course_cat_encoder = LabelEncoder()
        df['Course_Category_Encoded'] = course_cat_encoder.fit_transform(df['Course_Category'])
        
        df['Pass_Status'] = (df['Final Grade Numeric'] >= 1.0).astype(int)
        df['Grade_Points'] = df['Final Grade Numeric'] * df['Credits']
        
        all_features = numeric_features + ['Course_Category_Encoded', 'Pass_Status', 'Grade_Points']
        
        df[all_features] = self.scaler.fit_transform(df[all_features])
        
        return df[['Semester', 'student_id', 'Relative Term'] + all_features]
    
    def preprocess_performance_data(self, perf_df: pd.DataFrame) -> pd.DataFrame:
        df = perf_df.copy()
        
        numeric_cols = ['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg']
        df = self.clean_numeric_data(df, numeric_cols)
        
        df['Warning_Numeric'] = df['Warning'].map(self.warning_mapping).fillna(0)
        
        df['Level_Year'] = df['Level'].str.extract('(\d+)').astype(float).fillna(1)
        
        df['Pass_Rate'] = df['TC qua'] / (df['Reg'] + 1e-8)
        df['Debt_Rate'] = df['Debt'] / (df['Reg'] + 1e-8)
        df['Accumulation_Rate'] = df['Acc'] / (df['Relative Term'] * 20 + 1e-8)
        
        rate_cols = ['Pass_Rate', 'Debt_Rate', 'Accumulation_Rate']
        df = self.clean_numeric_data(df, rate_cols)
        
        performance_features = ['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg',
                              'Warning_Numeric', 'Level_Year', 'Pass_Rate', 
                              'Debt_Rate', 'Accumulation_Rate']
        
        df[performance_features] = self.scaler.fit_transform(df[performance_features])
        
        return df[['Semester', 'student_id', 'Relative Term'] + performance_features]

class StudentSequenceDataset(Dataset):
    def __init__(self, student_sequences: List[Dict], max_courses_per_semester: int = 15, 
                 max_semesters: int = 10):
        self.sequences = student_sequences
        self.max_courses = max_courses_per_semester
        self.max_semesters = max_semesters
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        course_features = torch.zeros(self.max_semesters, self.max_courses, 7)
        semester_features = torch.zeros(self.max_semesters, 11)
        course_masks = torch.zeros(self.max_semesters, self.max_courses)
        semester_masks = torch.zeros(self.max_semesters)
        
        for sem_idx, sem_data in enumerate(seq['semesters'][:self.max_semesters]):
            semester_features[sem_idx] = torch.tensor(sem_data['performance'], dtype=torch.float32)
            semester_masks[sem_idx] = 1.0
            
            courses = sem_data['courses'][:self.max_courses]
            for course_idx, course in enumerate(courses):
                course_features[sem_idx, course_idx] = torch.tensor(course, dtype=torch.float32)
                course_masks[sem_idx, course_idx] = 1.0
        
        target_gpa = seq['target_gpa']
        target_cpa = seq['target_cpa']
        
        return {
            'course_features': course_features,
            'semester_features': semester_features, 
            'course_masks': course_masks,
            'semester_masks': semester_masks,
            'target_gpa': torch.tensor(target_gpa, dtype=torch.float32),
            'target_cpa': torch.tensor(target_cpa, dtype=torch.float32)
        }

class CourseEncoder(nn.Module):
    def __init__(self, course_feature_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.course_embedding = nn.Sequential(
            nn.Linear(course_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, course_features, course_masks):
        batch_size, max_sems, max_courses, feat_dim = course_features.shape
        
        course_features_flat = course_features.view(-1, feat_dim)
        course_embeddings = self.course_embedding(course_features_flat)
        course_embeddings = course_embeddings.view(batch_size, max_sems, max_courses, -1)
        
        masked_embeddings = course_embeddings * course_masks.unsqueeze(-1)
        semester_course_sum = masked_embeddings.sum(dim=2)
        
        course_counts = course_masks.sum(dim=2, keepdim=True)
        semester_representations = semester_course_sum / (course_counts + 1e-8)
        
        return semester_representations

class SemesterTransformerEncoder(nn.Module):
    def __init__(self, semester_dim: int = 64, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(50, semester_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=semester_dim,
            nhead=num_heads,
            dim_feedforward=semester_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, semester_representations, semester_masks):
        seq_len = semester_representations.size(1)
        pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
        
        enhanced_representations = semester_representations + pos_enc
        
        attention_mask = ~semester_masks.bool()
        
        transformed = self.transformer(enhanced_representations, src_key_padding_mask=attention_mask)
        
        return transformed

class StudentPerformancePredictor(nn.Module):
    def __init__(self, course_feature_dim: int = 7, semester_feature_dim: int = 11, 
                 hidden_dim: int = 64, lstm_hidden: int = 128, num_heads: int = 8):
        super().__init__()
        
        self.course_encoder = CourseEncoder(course_feature_dim, hidden_dim)
        
        self.semester_feature_proj = nn.Linear(semester_feature_dim, hidden_dim)
        
        self.semester_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.transformer_encoder = SemesterTransformerEncoder(hidden_dim, num_heads, 2)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, course_features, semester_features, course_masks, semester_masks):
        course_representations = self.course_encoder(course_features, course_masks)
        
        semester_proj = self.semester_feature_proj(semester_features)
        
        fused_representations = self.semester_fusion(
            torch.cat([course_representations, semester_proj], dim=-1)
        )
        
        transformer_output = self.transformer_encoder(fused_representations, semester_masks)
        
        lstm_output, _ = self.lstm(transformer_output)
        
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output,
                                       key_padding_mask=~semester_masks.bool())
        
        valid_lengths = semester_masks.sum(dim=1).long()
        batch_indices = torch.arange(attn_output.size(0))
        last_valid_outputs = attn_output[batch_indices, valid_lengths - 1]
        
        predictions = self.predictor(last_valid_outputs)
        
        return predictions

class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, min_lr=1e-6)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            course_features = batch['course_features'].to(self.device)
            semester_features = batch['semester_features'].to(self.device)
            course_masks = batch['course_masks'].to(self.device)
            semester_masks = batch['semester_masks'].to(self.device)
            target_gpa = batch['target_gpa'].to(self.device)
            target_cpa = batch['target_cpa'].to(self.device)
            
            targets = torch.stack([target_gpa, target_cpa], dim=1)
            
            self.optimizer.zero_grad()
            predictions = self.model(course_features, semester_features, course_masks, semester_masks)
            
            if torch.isnan(predictions).any():
                print("Warning: NaN detected in predictions")
                continue
                
            loss = self.criterion(predictions, targets)
            
            if torch.isnan(loss):
                print("Warning: NaN detected in loss")
                continue
                
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                course_features = batch['course_features'].to(self.device)
                semester_features = batch['semester_features'].to(self.device)
                course_masks = batch['course_masks'].to(self.device)
                semester_masks = batch['semester_masks'].to(self.device)
                target_gpa = batch['target_gpa'].to(self.device)
                target_cpa = batch['target_cpa'].to(self.device)
                
                targets = torch.stack([target_gpa, target_cpa], dim=1)
                
                predictions = self.model(course_features, semester_features, course_masks, semester_masks)
                
                if torch.isnan(predictions).any():
                    continue
                    
                loss = self.criterion(predictions, targets)
                
                if torch.isnan(loss):
                    continue
                    
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        if not all_predictions:
            return {
                'loss': float('inf'),
                'gpa_mse': float('inf'), 'cpa_mse': float('inf'),
                'gpa_mae': float('inf'), 'cpa_mae': float('inf'),
                'gpa_r2': float('-inf'), 'cpa_r2': float('-inf')
            }
            
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        valid_mask = ~np.isnan(all_predictions).any(axis=1) & ~np.isnan(all_targets).any(axis=1)
        valid_predictions = all_predictions[valid_mask]
        valid_targets = all_targets[valid_mask]
        
        if len(valid_predictions) == 0:
            return {
                'loss': total_loss / len(dataloader),
                'gpa_mse': float('inf'), 'cpa_mse': float('inf'),
                'gpa_mae': float('inf'), 'cpa_mae': float('inf'),
                'gpa_r2': float('-inf'), 'cpa_r2': float('-inf')
            }
        
        gpa_mse = mean_squared_error(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_mse = mean_squared_error(valid_targets[:, 1], valid_predictions[:, 1])
        gpa_mae = mean_absolute_error(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_mae = mean_absolute_error(valid_targets[:, 1], valid_predictions[:, 1])
        gpa_r2 = r2_score(valid_targets[:, 0], valid_predictions[:, 0])
        cpa_r2 = r2_score(valid_targets[:, 1], valid_predictions[:, 1])
        
        return {
            'loss': total_loss / len(dataloader),
            'gpa_mse': gpa_mse, 'cpa_mse': cpa_mse,
            'gpa_mae': gpa_mae, 'cpa_mae': cpa_mae,
            'gpa_r2': gpa_r2, 'cpa_r2': cpa_r2
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
            print(f'  GPA - MSE: {val_metrics["gpa_mse"]:.4f}, MAE: {val_metrics["gpa_mae"]:.4f}, R2: {val_metrics["gpa_r2"]:.4f}')
            print(f'  CPA - MSE: {val_metrics["cpa_mse"]:.4f}, MAE: {val_metrics["cpa_mae"]:.4f}, R2: {val_metrics["cpa_r2"]:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        return train_losses, val_losses

def create_student_sequences(course_df: pd.DataFrame, perf_df: pd.DataFrame, 
                           processor: StudentPerformanceDataProcessor) -> List[Dict]:
    course_processed = processor.preprocess_course_data(course_df)
    perf_processed = processor.preprocess_performance_data(perf_df)
    sequences = []
    skipped_students = []
    
    for student_id in course_processed['student_id'].unique():
        student_courses = course_processed[course_processed['student_id'] == student_id]
        student_perf = perf_processed[perf_processed['student_id'] == student_id]
        
        if len(student_perf) < 2:
            skipped_students.append(student_id)
            continue
            
        semester_data = []
        has_null = False
        
        for _, perf_row in student_perf.iterrows():
            semester = perf_row['Semester']
            semester_courses = student_courses[student_courses['Semester'] == semester]
            
            if len(semester_courses) == 0:
                continue
                
            course_features = semester_courses[['Continuous Assessment Score', 'Exam Score', 'Credits',
                                             'Final Grade Numeric', 'Course_Category_Encoded', 
                                             'Pass_Status', 'Grade_Points']].values.tolist()
            
            performance_features = perf_row[['GPA', 'CPA', 'TC qua', 'Acc', 'Debt', 'Reg',
                                           'Warning_Numeric', 'Level_Year', 'Pass_Rate', 
                                           'Debt_Rate', 'Accumulation_Rate']].values.tolist()
            
            if np.isnan(course_features).any() or np.isnan(performance_features).any():
                has_null = True
                break
                
            semester_data.append({
                'semester': semester,
                'relative_term': perf_row['Relative Term'],
                'courses': course_features,
                'performance': performance_features
            })
        
        if has_null:
            skipped_students.append(student_id)
            continue
            
        semester_data.sort(key=lambda x: x['relative_term'])
        
        for i in range(len(semester_data) - 1):
            input_semesters = semester_data[:i+1]
            target_semester = semester_data[i+1]
            
            sequences.append({
                'student_id': student_id,
                'semesters': input_semesters,
                'target_gpa': target_semester['performance'][0],
                'target_cpa': target_semester['performance'][1]
            })
    
    print(f"\nSố sinh viên bị loại do dữ liệu null: {len(skipped_students)}")
    print(f"Danh sách sinh viên bị loại: {skipped_students}")
    
    return sequences

def main():
    parser = argparse.ArgumentParser(description='Student Performance Prediction Model')
    parser.add_argument('--course_csv', type=str, required=True, help='Path to course data CSV file')
    parser.add_argument('--performance_csv', type=str, required=True, help='Path to performance data CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--output_model', type=str, default='best_model.pth', help='Path to save the trained model')
    parser.add_argument('--output_plot', type=str, default='training_results.png', help='Path to save the training plot')
    
    args = parser.parse_args()
    
    print("=== MÔ HÌNH DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN ===")
    print("Architecture: Course Encoder + Transformer + LSTM + Attention")
    
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA không khả dụng! Kiểm tra:")
        print("1. Đã cài đặt NVIDIA GPU chưa?")
        print("2. Đã cài đặt NVIDIA Driver chưa?")
        print("3. Đã cài đặt CUDA Toolkit chưa?")
        print("4. Đã cài đặt PyTorch với CUDA support chưa?")
        print("\nChạy lệnh sau để cài đặt PyTorch với CUDA:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        device = torch.device('cpu')
    else:
        print("\nCUDA khả dụng!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Số GPU: {torch.cuda.device_count()}")
        print(f"GPU hiện tại: {torch.cuda.current_device()}")
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    
    print(f"\nSử dụng device: {device}")
    
    print("\nĐang tải dữ liệu...")
    print(f"Course CSV: {args.course_csv}")
    print(f"Performance CSV: {args.performance_csv}")
    
    course_df = pd.read_csv(args.course_csv)
    perf_df = pd.read_csv(args.performance_csv)
    
    print(f"Course data: {len(course_df)} records")
    print(f"Performance data: {len(perf_df)} records")
    
    print("\nĐang xử lý dữ liệu...")
    processor = StudentPerformanceDataProcessor()
    sequences = create_student_sequences(course_df, perf_df, processor)
    
    print(f"Đã tạo {len(sequences)} training sequences từ dữ liệu")
    
    train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
    train_sequences, val_sequences = train_test_split(train_sequences, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
    
    train_dataset = StudentSequenceDataset(train_sequences)
    val_dataset = StudentSequenceDataset(val_sequences)
    test_dataset = StudentSequenceDataset(test_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True if device.type == 'cuda' else False)
    
    print("\nĐang khởi tạo mô hình...")
    model = StudentPerformancePredictor()
    model = model.to(device)
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("\nKiểm tra model trên GPU:")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = ModelTrainer(model, device)
    
    print("\nBắt đầu training...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=args.epochs)
    
    print("\nĐang đánh giá mô hình trên test set...")
    trainer.model.load_state_dict(torch.load(args.output_model))
    test_metrics = trainer.evaluate(test_loader)
    
    print("\n" + "="*50)
    print("KẾT QUẢ CUỐI CÙNG")
    print("="*50)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"\nGPA Prediction:")
    print(f"  - MSE: {test_metrics['gpa_mse']:.4f}")
    print(f"  - MAE: {test_metrics['gpa_mae']:.4f}")
    print(f"  - R²: {test_metrics['gpa_r2']:.4f}")
    print(f"\nCPA Prediction:")
    print(f"  - MSE: {test_metrics['cpa_mse']:.4f}")
    print(f"  - MAE: {test_metrics['cpa_mae']:.4f}")
    print(f"  - R²: {test_metrics['cpa_r2']:.4f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    with torch.no_grad():
        trainer.model.eval()
        gpa_predictions = []
        gpa_targets = []
        cpa_predictions = []
        cpa_targets = []
        
        for batch in test_loader:
            course_features = batch['course_features'].to(device)
            semester_features = batch['semester_features'].to(device)
            course_masks = batch['course_masks'].to(device)
            semester_masks = batch['semester_masks'].to(device)
            target_gpa = batch['target_gpa']
            target_cpa = batch['target_cpa']
            
            predictions = trainer.model(course_features, semester_features, course_masks, semester_masks)
            
            gpa_predictions.extend(predictions[:, 0].cpu().numpy())
            gpa_targets.extend(target_gpa.numpy())
            cpa_predictions.extend(predictions[:, 1].cpu().numpy())
            cpa_targets.extend(target_cpa.numpy())
    
    plt.subplot(1, 3, 2)
    plt.scatter(gpa_targets, gpa_predictions, alpha=0.6, color='blue')
    plt.plot([0, 4], [0, 4], 'r--', alpha=0.8)
    plt.xlabel('Actual GPA')
    plt.ylabel('Predicted GPA')
    plt.title(f'GPA Prediction (R²={test_metrics["gpa_r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(cpa_targets, cpa_predictions, alpha=0.6, color='green')
    plt.plot([0, 4], [0, 4], 'r--', alpha=0.8)
    plt.xlabel('Actual CPA')
    plt.ylabel('Predicted CPA')
    plt.title(f'CPA Prediction (R²={test_metrics["cpa_r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300, bbox_inches='tight')
    plt.show()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("\nThông tin GPU sau khi training:")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print(f"\nĐã lưu kết quả vào '{args.output_plot}' và model weights vào '{args.output_model}'")

if __name__ == "__main__":
    main() 