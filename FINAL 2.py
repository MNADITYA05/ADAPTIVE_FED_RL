import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptiveArchitectureNet(nn.Module):
    def __init__(self, input_size, output_size=1, data_size=1000, complexity_factor=1.5):
        super(AdaptiveArchitectureNet, self).__init__()
        
        self.input_size = input_size
        
        if data_size < 500:
            hidden_size = max(64, input_size * 3)
            num_layers = 4
            dropout_rate = 0.15
        elif data_size < 2000:
            hidden_size = max(128, input_size * 4)
            num_layers = 5
            dropout_rate = 0.2
        elif data_size < 5000:
            hidden_size = max(256, input_size * 5)
            num_layers = 5
            dropout_rate = 0.25
        else:
            hidden_size = max(512, input_size * 6)
            num_layers = 6
            dropout_rate = 0.3
        
        hidden_size = int(hidden_size * complexity_factor)
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers - 1):
            next_size = hidden_size // (2 ** i) if i > 0 else hidden_size
            next_size = max(next_size, 64)
            
            layers.extend([
                nn.Linear(current_size, next_size),
                nn.BatchNorm1d(next_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_size = next_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
        
    def forward(self, x):
        return self.network(x)

class DatasetAnalyzer:
    def __init__(self):
        self.stats = {}
        
    def analyze(self, y_data):
        self.stats = {
            'mean': np.mean(y_data),
            'std': np.std(y_data),
            'min': np.min(y_data),
            'max': np.max(y_data),
            'range': np.max(y_data) - np.min(y_data),
            'cv': np.std(y_data) / np.mean(y_data) if np.mean(y_data) != 0 else 1.0
        }
        return self.stats
    
    def get_performance_thresholds(self, historical_rmse, historical_mae):
        if len(historical_rmse) > 5:
            rmse_baseline = np.median(historical_rmse[-5:])
            mae_baseline = np.median(historical_mae[-5:])
        else:
            rmse_baseline = self.stats.get('std', 5.0)
            mae_baseline = self.stats.get('std', 4.0)
        
        return {
            'rmse': {
                'excellent': rmse_baseline * 0.8,
                'good': rmse_baseline * 1.0,
                'acceptable': rmse_baseline * 1.2,
                'poor': rmse_baseline * 1.5,
                'catastrophic': rmse_baseline * 2.0
            },
            'mae': {
                'excellent': mae_baseline * 0.8,
                'good': mae_baseline * 1.0,
                'acceptable': mae_baseline * 1.2,
                'poor': mae_baseline * 1.5,
                'catastrophic': mae_baseline * 2.0
            }
        }

class AdaptivePerformanceFedPruning:
    def __init__(self, input_size, num_clients=5, lr=0.01, all_y_data=None, data_sizes=None):
        self.num_clients = num_clients
        self.input_size = input_size
        self.lr = lr
        
        self.analyzer = DatasetAnalyzer()
        if all_y_data is not None:
            self.dataset_stats = self.analyzer.analyze(all_y_data)
        else:
            self.dataset_stats = {}
        
        avg_data_size = np.mean(data_sizes) if data_sizes else 1000
        self.global_model = AdaptiveArchitectureNet(input_size, data_size=avg_data_size).to(device)
        self.client_models = [AdaptiveArchitectureNet(input_size, data_size=size).to(device) 
                             for size in (data_sizes or [avg_data_size] * num_clients)]
        self.client_masks = [None for _ in range(num_clients)]
        self.client_data_sizes = data_sizes or [avg_data_size] * num_clients
        self.global_mask = None
        
        self.original_model_size = self.global_model.get_model_size_mb()
        self.original_param_count = self.global_model.count_parameters()
        
        self.current_target_sparsity = 0.5
        self.max_target_sparsity = 0.85
        self.min_target_sparsity = 0.3
        self.sparsity_increment = 0.1
        self.performance_patience = 5
        self.pruning_cooldown = 1
        self.last_pruning_round = -10
        
        self.sparsity_schedule = {
            10: 0.4,
            20: 0.55,
            30: 0.65,
            40: 0.75,
            50: 0.8
        }
        
        self.round_count = 0
        
        self.client_performance_history = {
            'rmse': [[] for _ in range(num_clients)],
            'mae': [[] for _ in range(num_clients)]
        }
        self.global_performance_history = {
            'rmse': [],
            'mae': [],
            'model_size_reduction': [],
            'target_sparsity': []
        }
        self.performance_thresholds = None
        
    def get_schedule_target_sparsity(self, round_num):
        for round_threshold in sorted(self.sparsity_schedule.keys()):
            if round_num <= round_threshold:
                return self.sparsity_schedule[round_threshold]
        return self.max_target_sparsity
        
    def update_performance_thresholds(self):
        all_rmse = []
        all_mae = []
        for client_history in self.client_performance_history['rmse']:
            all_rmse.extend(client_history)
        for client_history in self.client_performance_history['mae']:
            all_mae.extend(client_history)
        
        if len(all_rmse) > 0:
            self.performance_thresholds = self.analyzer.get_performance_thresholds(all_rmse, all_mae)
    
    def assess_performance_trend(self, rmse_history, mae_history, window=3):
        if len(rmse_history) < window:
            return "stable"
        
        recent_rmse = rmse_history[-window:]
        older_rmse = rmse_history[-2*window:-window] if len(rmse_history) >= 2*window else rmse_history[:-window]
        
        if len(older_rmse) == 0:
            return "stable"
        
        recent_avg = np.mean(recent_rmse)
        older_avg = np.mean(older_rmse)
        
        improvement_ratio = (older_avg - recent_avg) / max(older_avg, 1e-6)
        
        if improvement_ratio > 0.05:
            return "improving"
        elif improvement_ratio < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def should_prune_adaptive(self, round_num, client_id, current_rmse, current_mae):
        if round_num < 3:
            return False
        
        current_sparsity = self.calculate_sparsity_consistent(self.client_models[client_id])
        schedule_target = self.get_schedule_target_sparsity(round_num)
        
        if current_sparsity < schedule_target * 0.8:
            return True
        
        if round_num - self.last_pruning_round < self.pruning_cooldown:
            return False
        
        if self.performance_thresholds is None:
            self.update_performance_thresholds()
            return round_num >= 5
        
        performance_level = "good"
        if current_rmse <= self.performance_thresholds['rmse']['excellent']:
            performance_level = "excellent"
        elif current_rmse <= self.performance_thresholds['rmse']['good']:
            performance_level = "good"
        elif current_rmse <= self.performance_thresholds['rmse']['acceptable']:
            performance_level = "acceptable"
        elif current_rmse <= self.performance_thresholds['rmse']['poor']:
            performance_level = "poor"
        else:
            performance_level = "catastrophic"
        
        if performance_level in ["excellent", "good", "acceptable"] or current_sparsity < 0.5:
            return True
        
        return False
    
    def calculate_adaptive_pruning_ratio(self, round_num, client_id, current_rmse, current_mae, current_sparsity):
        schedule_target = self.get_schedule_target_sparsity(round_num)
        sparsity_gap = max(0, schedule_target - current_sparsity)
        
        if round_num <= 15:
            base_ratio = 0.3
        elif round_num <= 30:
            base_ratio = 0.25
        elif round_num <= 50:
            base_ratio = 0.2
        else:
            base_ratio = 0.15
        
        if sparsity_gap > 0.3:
            gap_factor = 1.5
        elif sparsity_gap > 0.2:
            gap_factor = 1.3
        elif sparsity_gap > 0.1:
            gap_factor = 1.1
        else:
            gap_factor = 1.0
        
        data_size = self.client_data_sizes[client_id]
        if data_size > 400:
            size_factor = 1.1
        elif data_size > 200:
            size_factor = 1.0
        else:
            size_factor = 0.9
        
        final_ratio = base_ratio * gap_factor * size_factor
        return np.clip(final_ratio, 0.15, 0.6)
    
    def adapt_target_sparsity(self, round_num):
        schedule_target = self.get_schedule_target_sparsity(round_num)
        self.current_target_sparsity = schedule_target
        
        if round_num < 10 or len(self.global_performance_history['rmse']) < 5:
            return
        
        recent_performance = self.global_performance_history['rmse'][-3:]
        older_performance = self.global_performance_history['rmse'][-6:-3]
        
        if len(older_performance) < 3:
            return
        
        recent_avg = np.mean(recent_performance)
        older_avg = np.mean(older_performance)
        
        performance_change = (recent_avg - older_avg) / max(older_avg, 1e-6)
        
        current_global_sparsity = self.calculate_sparsity_consistent(self.global_model)
        
        if performance_change < -0.2:
            self.current_target_sparsity = max(
                schedule_target * 0.9,
                min(self.current_target_sparsity - self.sparsity_increment * 0.5, current_global_sparsity)
            )
        elif performance_change > 0.05 and current_global_sparsity >= self.current_target_sparsity * 0.9:
            self.current_target_sparsity = min(
                self.max_target_sparsity,
                max(schedule_target, self.current_target_sparsity + self.sparsity_increment)
            )
        
        self.global_performance_history['target_sparsity'].append(self.current_target_sparsity)
    
    def calculate_adaptive_learning_rate(self, round_num):
        if round_num <= 20:
            factor = 1.0
        elif round_num <= 40:
            factor = 0.8
        elif round_num <= 60:
            factor = 0.5
        elif round_num <= 80:
            factor = 0.2
        else:
            factor = 0.1
        return self.lr * factor
    
    def calculate_sparsity_consistent(self, model):
        with torch.no_grad():
            total_weights = 0
            zero_weights = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.dtype.is_floating_point and len(param.shape) > 1:
                    total_weights += param.numel()
                    zero_weights += (torch.abs(param) < 1e-6).sum().item()
            return zero_weights / total_weights if total_weights > 0 else 0
    
    def adaptive_magnitude_pruning(self, model, pruning_ratio, client_id):
        mask = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype.is_floating_point and len(param.shape) > 1:
                importance = torch.abs(param.data)
                flat_importance = importance.flatten()
                
                if len(flat_importance) > 0:
                    k_thresh = int(len(flat_importance) * pruning_ratio)
                    if k_thresh < len(flat_importance):
                        threshold = torch.kthvalue(flat_importance, k_thresh + 1).values
                        param_mask = (importance > threshold).float()
                    else:
                        param_mask = torch.zeros_like(param).float()
                    
                    d_out, d_in = param.shape[0], param.shape[1]
                    min_connectivity = max(1, int(0.01 * d_out * d_in))
                    
                    if param_mask.sum() < min_connectivity:
                        n_keep = max(min_connectivity, 1)
                        if n_keep < len(flat_importance):
                            sorted_vals, _ = torch.sort(flat_importance, descending=True)
                            threshold_adjusted = sorted_vals[n_keep-1]
                            param_mask = (importance >= threshold_adjusted).float()
                        else:
                            param_mask = torch.ones_like(param).float()
                    
                    mask[name] = param_mask
                else:
                    mask[name] = torch.ones_like(param).float()
            else:
                mask[name] = torch.ones_like(param).float()
        
        return mask
    
    def apply_mask(self, model, mask):
        if mask is None:
            return
        for name, param in model.named_parameters():
            if name in mask and param.requires_grad and param.dtype.is_floating_point:
                param.data *= mask[name].to(param.device)
    
    def calculate_data_proportional_weights(self, client_data_sizes):
        total_data_size = sum(client_data_sizes)
        if total_data_size == 0:
            return [1.0 / len(client_data_sizes)] * len(client_data_sizes)
        
        weights = [size / total_data_size for size in client_data_sizes]
        return weights
    
    def adaptive_sparsity_preserving_aggregation(self, client_models, client_masks, client_data_sizes):
        global_dict = self.global_model.state_dict()
        weights = self.calculate_data_proportional_weights(client_data_sizes)
        
        min_overlap_threshold = max(0.15, 1.0 - self.current_target_sparsity)
        
        for key in global_dict.keys():
            if global_dict[key].dtype.is_floating_point and len(global_dict[key].shape) > 1:
                weighted_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
                total_weight = torch.zeros_like(global_dict[key], dtype=torch.float32)
                overlap_count = torch.zeros_like(global_dict[key], dtype=torch.float32)
                
                for i, (model, mask, weight) in enumerate(zip(client_models, client_masks, weights)):
                    client_dict = model.state_dict()
                    client_param = client_dict[key].float()
                    
                    if mask is not None and key in mask:
                        mask_tensor = mask[key].float()
                        masked_param = client_param * mask_tensor
                        weighted_sum += masked_param * weight
                        total_weight += mask_tensor * weight
                        overlap_count += mask_tensor
                    else:
                        weighted_sum += client_param * weight
                        total_weight += torch.ones_like(client_param) * weight
                        overlap_count += torch.ones_like(client_param)
                
                overlap_ratio = overlap_count / len(client_models)
                sufficient_overlap = overlap_ratio >= min_overlap_threshold
                
                total_weight = torch.clamp(total_weight, min=1e-8)
                
                aggregated_param = weighted_sum / total_weight
                global_dict[key] = torch.where(
                    sufficient_overlap,
                    aggregated_param,
                    global_dict[key].float()
                ).to(global_dict[key].dtype)
            
            elif global_dict[key].dtype.is_floating_point:
                param_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
                count = 0
                
                for i, (model, mask, weight) in enumerate(zip(client_models, client_masks, weights)):
                    client_dict = model.state_dict()
                    param_sum += client_dict[key].float() * weight
                    count += weight
                
                if count > 0:
                    global_dict[key] = (param_sum / count).to(global_dict[key].dtype)
            
            else:
                param_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
                count = 0
                
                for i, (model, mask, weight) in enumerate(zip(client_models, client_masks, weights)):
                    client_dict = model.state_dict()
                    param_sum += client_dict[key].float() * weight
                    count += weight
                
                if count > 0:
                    global_dict[key] = (param_sum / count).to(global_dict[key].dtype)
        
        self.global_model.load_state_dict(global_dict)
    
    def adaptive_global_pruning(self):
        current_global_sparsity = self.calculate_sparsity_consistent(self.global_model)
        schedule_target = self.get_schedule_target_sparsity(self.round_count)
        
        if current_global_sparsity < schedule_target * 0.95:
            target_pruning = min(schedule_target, current_global_sparsity + 0.1)
            
            all_weights = []
            weight_info = []
            
            for name, param in self.global_model.named_parameters():
                if param.requires_grad and param.dtype.is_floating_point and len(param.shape) > 1:
                    flat_weights = torch.abs(param.data).flatten()
                    all_weights.append(flat_weights)
                    weight_info.append((name, param.shape, len(flat_weights)))
            
            if len(all_weights) == 0:
                return
            
            all_weights_tensor = torch.cat(all_weights)
            total_params = len(all_weights_tensor)
            k_thresh = int(total_params * target_pruning)
            
            if k_thresh > 0 and k_thresh < total_params:
                threshold = torch.kthvalue(all_weights_tensor, k_thresh + 1).values
                
                global_mask = {}
                
                for name, shape, length in weight_info:
                    param = dict(self.global_model.named_parameters())[name]
                    importance = torch.abs(param.data)
                    mask = (importance > threshold).float()
                    
                    d_out, d_in = shape[0], shape[1]
                    min_connectivity = max(1, int(0.01 * d_out * d_in))
                    
                    if mask.sum() < min_connectivity:
                        flat_importance = importance.flatten()
                        sorted_vals, _ = torch.sort(flat_importance, descending=True)
                        threshold_adjusted = sorted_vals[min_connectivity-1]
                        mask = (importance >= threshold_adjusted).float()
                    
                    global_mask[name] = mask
                    param.data *= mask
                
                self.global_mask = global_mask
    
    def calculate_model_size_reduction(self):
        sparsity = self.calculate_sparsity_consistent(self.global_model)
        return sparsity * 100
    
    def train_client(self, client_id, train_loader, val_loader, epochs=20, round_num=0):
        model = self.client_models[client_id]
        model.load_state_dict(self.global_model.state_dict())
        
        if self.global_mask is not None:
            self.apply_mask(model, self.global_mask)
            self.client_masks[client_id] = copy.deepcopy(self.global_mask)
        
        adaptive_lr = self.calculate_adaptive_learning_rate(round_num)
        
        optimizer = optim.AdamW(model.parameters(), lr=adaptive_lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if self.client_masks[client_id] is not None:
                    self.apply_mask(model, self.client_masks[client_id])
            
            scheduler.step()
        
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_predictions.extend(output.cpu().numpy().flatten())
                val_targets.extend(target.cpu().numpy().flatten())
        
        rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        mae = mean_absolute_error(val_targets, val_predictions)
        avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        self.client_performance_history['rmse'][client_id].append(rmse)
        self.client_performance_history['mae'][client_id].append(mae)
        
        should_prune = self.should_prune_adaptive(round_num, client_id, rmse, mae)
        
        if should_prune:
            current_sparsity = self.calculate_sparsity_consistent(model)
            pruning_ratio = self.calculate_adaptive_pruning_ratio(round_num, client_id, rmse, mae, current_sparsity)
            
            mask = self.adaptive_magnitude_pruning(model, pruning_ratio, client_id)
            self.apply_mask(model, mask)
            self.client_masks[client_id] = mask
            self.last_pruning_round = round_num
            
            model.train()
            fine_tune_lr = adaptive_lr * 0.3
            fine_tune_optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=1e-5)
            
            fine_tune_epochs = min(8, max(3, int(pruning_ratio * 15)))
            for ft_epoch in range(fine_tune_epochs):
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    fine_tune_optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    fine_tune_optimizer.step()
                    self.apply_mask(model, mask)
        
        sparsity = self.calculate_sparsity_consistent(model)
        return rmse, mae, avg_loss, sparsity, self.client_data_sizes[client_id]
    
    def federated_round(self, client_loaders, val_loaders, round_num):
        rmses = []
        maes = []
        losses = []
        sparsities = []
        data_sizes = []
        
        num_clients = len(client_loaders)
        participation_prob = max(0.6, 1.0 - round_num * 0.002)
        num_selected = max(2, int(num_clients * participation_prob))
        selected_clients = random.sample(range(num_clients), num_selected)
        
        for client_id in selected_clients:
            if len(client_loaders) > client_id and len(val_loaders) > client_id:
                rmse, mae, loss, sparsity, data_size = self.train_client(
                    client_id, 
                    client_loaders[client_id], 
                    val_loaders[client_id],
                    epochs=20,
                    round_num=round_num
                )
                rmses.append(rmse)
                maes.append(mae)
                losses.append(loss)
                sparsities.append(sparsity)
                data_sizes.append(data_size)
        
        if len(rmses) > 0:
            active_models = [self.client_models[i] for i in selected_clients]
            active_masks = [self.client_masks[i] for i in selected_clients]
            active_data_sizes = [self.client_data_sizes[i] for i in selected_clients]
            
            self.adaptive_sparsity_preserving_aggregation(active_models, active_masks, active_data_sizes)
            
            if round_num % 2 == 0:
                self.adaptive_global_pruning()
            
            avg_rmse = np.mean(rmses)
            avg_mae = np.mean(maes)
            model_size_reduction = self.calculate_model_size_reduction()
            
            self.global_performance_history['rmse'].append(avg_rmse)
            self.global_performance_history['mae'].append(avg_mae)
            self.global_performance_history['model_size_reduction'].append(model_size_reduction)
            
            self.adapt_target_sparsity(round_num)
            
            if round_num % 5 == 0:
                self.update_performance_thresholds()
        
        self.round_count += 1
        
        return np.mean(rmses) if rmses else float('inf'), np.mean(maes) if maes else float('inf'), np.mean(losses) if losses else 0, np.mean(sparsities) if sparsities else 0

def load_real_soybean_data():
    try:
        df = pd.read_csv('/kaggle/input/nitkuruk/soybean_samples.csv')
        
        try:
            loc_df = pd.read_csv('/kaggle/input/nitkuruk/Soybeans_Loc_ID.csv')
        except:
            loc_df = None
        
        return df, loc_df
        
    except FileNotFoundError as e:
        print(f"Required data files not found: {e}")
        raise FileNotFoundError("Cannot proceed without real data files")

def preprocess_real_data(df, loc_df=None):
    if 'yield' not in df.columns:
        yield_columns = [col for col in df.columns if 'yield' in col.lower()]
        if yield_columns:
            df = df.rename(columns={yield_columns[0]: 'yield'})
        else:
            raise ValueError("No yield column found in the dataset")
    
    df = df.dropna(subset=['yield'])
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'yield' in numeric_columns:
        numeric_columns.remove('yield')
    
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    valid_yield_mask = (df['yield'] >= 5) & (df['yield'] <= 100) & (df['yield'].notna())
    df = df[valid_yield_mask]
    
    feature_cols = numeric_columns
    
    if loc_df is not None and 'loc_ID' in df.columns and 'ID_loc' in loc_df.columns:
        df = df.merge(loc_df, left_on='loc_ID', right_on='ID_loc', how='left')
        if 'State' in df.columns:
            state_dummies = pd.get_dummies(df['State'], prefix='State')
            df = pd.concat([df, state_dummies], axis=1)
            feature_cols.extend(state_dummies.columns.tolist())
    
    X = df[feature_cols].values
    y = df['yield'].values
    
    metadata_cols = []
    if 'loc_ID' in df.columns:
        metadata_cols.append('loc_ID')
    if 'year' in df.columns:
        metadata_cols.append('year')
    elif 'Year' in df.columns:
        metadata_cols.append('Year')
    
    if metadata_cols:
        metadata = df[metadata_cols].values
    else:
        metadata = np.column_stack([
            np.arange(len(df)),
            np.full(len(df), 2020)
        ])
    
    X = np.nan_to_num(X, nan=0.0)
    
    return X, y, metadata, feature_cols

def create_federated_clients_real(X, y, metadata, num_clients=5, min_samples_per_client=50):
    if len(X) < num_clients * min_samples_per_client:
        num_clients = max(1, len(X) // min_samples_per_client)
    
    if metadata.shape[1] > 0 and len(np.unique(metadata[:, 0])) > num_clients:
        unique_locations = np.unique(metadata[:, 0])
        np.random.shuffle(unique_locations)
        locs_per_client = len(unique_locations) // num_clients
        
        client_data = []
        client_sizes = []
        
        for client_id in range(num_clients):
            start_idx = client_id * locs_per_client
            end_idx = (client_id + 1) * locs_per_client if client_id < num_clients - 1 else len(unique_locations)
            client_locations = unique_locations[start_idx:end_idx]
            
            location_mask = np.isin(metadata[:, 0], client_locations)
            client_X = X[location_mask]
            client_y = y[location_mask]
            
            if len(client_X) < min_samples_per_client:
                continue
                
            X_train, X_val, y_train, y_val = train_test_split(
                client_X, client_y, test_size=0.3, random_state=42 + client_id
            )
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train).unsqueeze(1)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val).unsqueeze(1)
            )
            
            batch_size = min(32, max(8, len(X_train) // 4))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            client_data.append((train_loader, val_loader))
            client_sizes.append(len(X_train))
    
    else:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        samples_per_client = len(X) // num_clients
        
        client_data = []
        client_sizes = []
        
        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            end_idx = (client_id + 1) * samples_per_client if client_id < num_clients - 1 else len(X)
            
            client_indices = indices[start_idx:end_idx]
            client_X = X[client_indices]
            client_y = y[client_indices]
            
            if len(client_X) < min_samples_per_client:
                continue
            
            X_train, X_val, y_train, y_val = train_test_split(
                client_X, client_y, test_size=0.3, random_state=42 + client_id
            )
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train).unsqueeze(1)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val).unsqueeze(1)
            )
            
            batch_size = min(32, max(8, len(X_train) // 4))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            client_data.append((train_loader, val_loader))
            client_sizes.append(len(X_train))
    
    return client_data, client_sizes

def run_adaptive_performance_experiment():
    df, loc_df = load_real_soybean_data()
    
    X, y, metadata, feature_cols = preprocess_real_data(df, loc_df)
    
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    y_scaler = StandardScaler()
    y_normalized = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.2, random_state=42, stratify=None
    )
    
    train_metadata = metadata[:len(X_train_all)]
    
    num_clients = min(8, max(3, len(X_train_all) // 200))
    client_data, client_sizes = create_federated_clients_real(X_train_all, y_train_all, train_metadata, 
                                                              num_clients=num_clients)
    
    if len(client_data) == 0:
        raise ValueError("No valid client data created. Check data quality and size.")
    
    input_size = X_normalized.shape[1]
    fed_system = AdaptivePerformanceFedPruning(
        input_size=input_size,
        num_clients=len(client_data),
        lr=0.01,
        all_y_data=y_train_all,
        data_sizes=client_sizes
    )
    
    train_loaders = [data[0] for data in client_data]
    val_loaders = [data[1] for data in client_data]
    
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    num_rounds = 80
    results = {
        'train_rmse': [],
        'val_rmse': [],
        'test_rmse': [],
        'test_mae': [],
        'loss': [],
        'sparsity': [],
        'communication_cost': [],
        'target_sparsity': [],
        'model_size_reduction': []
    }
    
    best_test_rmse = float('inf')
    best_test_mae = float('inf')
    best_round = 0
    patience_counter = 0
    patience_limit = 30
    
    for round_num in range(num_rounds):
        avg_train_rmse, avg_train_mae, avg_loss, avg_sparsity = fed_system.federated_round(
            train_loaders, val_loaders, round_num
        )
        
        if avg_train_rmse == float('inf'):
            break
        
        fed_system.global_model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = fed_system.global_model(data)
                test_predictions.extend(output.cpu().numpy().flatten())
                test_targets.extend(target.cpu().numpy().flatten())
        
        test_predictions_orig = y_scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
        test_targets_orig = y_scaler.inverse_transform(np.array(test_targets).reshape(-1, 1)).flatten()
        
        test_rmse = np.sqrt(mean_squared_error(test_targets_orig, test_predictions_orig))
        test_mae = mean_absolute_error(test_targets_orig, test_predictions_orig)
        test_r2 = r2_score(test_targets_orig, test_predictions_orig)
        
        global_sparsity = fed_system.calculate_sparsity_consistent(fed_system.global_model)
        communication_cost = (1 - global_sparsity) * 100
        model_size_reduction = fed_system.calculate_model_size_reduction()
        compression_ratio = 1 / (1 - global_sparsity) if global_sparsity < 1 else float('inf')
        
        results['train_rmse'].append(avg_train_rmse)
        results['val_rmse'].append(avg_train_rmse)
        results['test_rmse'].append(test_rmse)
        results['test_mae'].append(test_mae)
        results['loss'].append(avg_loss)
        results['sparsity'].append(global_sparsity)
        results['communication_cost'].append(communication_cost)
        results['target_sparsity'].append(fed_system.current_target_sparsity)
        results['model_size_reduction'].append(model_size_reduction)
        
        if global_sparsity > 0.6 and (test_rmse < best_test_rmse or test_mae < best_test_mae):
            best_test_rmse = min(best_test_rmse, test_rmse)
            best_test_mae = min(best_test_mae, test_mae)
            best_round = round_num
            patience_counter = 0
            
            torch.save({
                'model_state_dict': fed_system.global_model.state_dict(),
                'round': round_num,
                'rmse': test_rmse,
                'mae': test_mae,
                'sparsity': global_sparsity,
                'target_sparsity': fed_system.current_target_sparsity,
                'model_size_reduction': model_size_reduction
            }, 'best_adaptive_performance_model.pth')
        else:
            patience_counter += 1
        
        if round_num % 5 == 0 or round_num == num_rounds - 1:
            print(f"Round {round_num + 1:3d}: "
                  f"RMSE: {test_rmse:.4f} | "
                  f"MAE: {test_mae:.4f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Sparsity: {global_sparsity:.3f} | "
                  f"Target: {fed_system.current_target_sparsity:.3f} | "
                  f"Compression: {compression_ratio:.1f}x | "
                  f"R²: {test_r2:.4f}")
        
        if round_num > 25 and patience_counter >= patience_limit:
            print(f"Early stopping at round {round_num + 1}")
            break
    
    try:
        checkpoint = torch.load('best_adaptive_performance_model.pth')
        fed_system.global_model.load_state_dict(checkpoint['model_state_dict'])
    except:
        pass
    
    fed_system.global_model.eval()
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = fed_system.global_model(data)
            final_predictions.extend(output.cpu().numpy().flatten())
            final_targets.extend(target.cpu().numpy().flatten())
    
    final_predictions_orig = y_scaler.inverse_transform(np.array(final_predictions).reshape(-1, 1)).flatten()
    final_targets_orig = y_scaler.inverse_transform(np.array(final_targets).reshape(-1, 1)).flatten()
    
    final_rmse = np.sqrt(mean_squared_error(final_targets_orig, final_predictions_orig))
    final_mae = mean_absolute_error(final_targets_orig, final_predictions_orig)
    final_r2 = r2_score(final_targets_orig, final_predictions_orig)
    final_mape = np.mean(np.abs((final_targets_orig - final_predictions_orig) / 
                               np.maximum(final_targets_orig, 1e-6))) * 100
    
    final_sparsity = fed_system.calculate_sparsity_consistent(fed_system.global_model)
    compression_ratio = 1 / (1 - final_sparsity) if final_sparsity < 1 else float('inf')
    communication_savings = final_sparsity * 100
    final_model_size_reduction = fed_system.calculate_model_size_reduction()
    
    plt.figure(figsize=(20, 16))
    
    plt.subplot(4, 3, 1)
    plt.plot(results['train_rmse'], label='Train RMSE', color='blue', alpha=0.7)
    plt.plot(results['test_rmse'], label='Test RMSE', color='red', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('RMSE')
    plt.title('RMSE Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 2)
    plt.plot(results['test_mae'], label='Test MAE', color='green', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('MAE')
    plt.title('MAE Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 3)
    plt.plot(results['sparsity'], color='purple', linewidth=2, label='Actual Global')
    plt.plot(results['target_sparsity'], color='orange', linewidth=2, linestyle='--', label='Adaptive Target')
    plt.xlabel('Round')
    plt.ylabel('Sparsity')
    plt.title('Adaptive Sparsity Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 4)
    compression_ratios = [1/(1-s) if s < 0.99 else 100 for s in results['sparsity']]
    plt.plot(compression_ratios, color='magenta', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Compression Ratio')
    plt.title('Adaptive Compression Evolution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 5)
    plt.plot(results['loss'], color='brown', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 6)
    plt.plot(results['model_size_reduction'], color='red', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Model Size Reduction (%)')
    plt.title('Model Size Reduction')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 7)
    plt.plot(results['communication_cost'], color='cyan', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Communication Cost (%)')
    plt.title('Communication Efficiency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 8)
    plt.scatter(final_targets_orig, final_predictions_orig, alpha=0.6, s=20)
    min_val = min(min(final_targets_orig), min(final_predictions_orig))
    max_val = max(max(final_targets_orig), max(final_predictions_orig))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'Predictions vs Actual (R² = {final_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 9)
    residuals = final_targets_orig - final_predictions_orig
    plt.scatter(final_predictions_orig, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Yield')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 10)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 11)
    plt.scatter(results['sparsity'], results['test_rmse'], alpha=0.6, color='red', s=30, label='RMSE')
    plt.scatter(results['sparsity'], results['test_mae'], alpha=0.6, color='green', s=30, label='MAE')
    plt.xlabel('Global Sparsity')
    plt.ylabel('Error')
    plt.title('Adaptive Sparsity vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 12)
    plt.plot(results['target_sparsity'], color='orange', linewidth=2, label='Adaptive Target')
    plt.plot(results['sparsity'], color='purple', linewidth=2, alpha=0.7, label='Achieved')
    plt.xlabel('Round')
    plt.ylabel('Sparsity')
    plt.title('Target vs Achieved Sparsity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_performance_federated_pruning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'final_r2': final_r2,
        'final_mape': final_mape,
        'final_sparsity': final_sparsity,
        'target_sparsity': fed_system.current_target_sparsity,
        'best_rmse': best_test_rmse,
        'best_mae': best_test_mae,
        'compression_ratio': compression_ratio,
        'communication_savings': communication_savings,
        'model_size_reduction': final_model_size_reduction,
        'results_history': results,
        'total_rounds': round_num + 1,
        'num_clients': len(client_data),
        'input_features': input_size,
        'train_samples': len(X_train_all),
        'test_samples': len(X_test),
        'dataset_stats': fed_system.dataset_stats,
        'original_model_size_mb': fed_system.original_model_size,
        'original_param_count': fed_system.original_param_count,
        'adaptive_settings': {
            'max_target_sparsity': fed_system.max_target_sparsity,
            'min_target_sparsity': fed_system.min_target_sparsity,
            'sparsity_increment': fed_system.sparsity_increment
        }
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    try:
        results = run_adaptive_performance_experiment()
        
        print(f"Final RMSE: {results['final_rmse']:.4f}")
        print(f"Final MAE: {results['final_mae']:.4f}")
        print(f"Final R²: {results['final_r2']:.4f}")
        print(f"Best RMSE: {results['best_rmse']:.4f}")
        print(f"Best MAE: {results['best_mae']:.4f}")
        print(f"Model Compression: {results['compression_ratio']:.1f}x")
        print(f"Communication Savings: {results['communication_savings']:.1f}%")
        print(f"Model Size Reduction: {results['model_size_reduction']:.1f}%")
        print(f"Final Target Sparsity: {results['target_sparsity']:.3f}")
        print(f"Achieved Sparsity: {results['final_sparsity']:.3f}")
        print(f"Original Model Size: {results['original_model_size_mb']:.2f} MB")
        print(f"Original Parameters: {results['original_param_count']:,}")
        print(f"Clients: {results['num_clients']}")
        print(f"Features: {results['input_features']}")
        print(f"Train Samples: {results['train_samples']}")
        print(f"Test Samples: {results['test_samples']}")
        print(f"Total Rounds: {results['total_rounds']}")
        print(f"Adaptive Range: {results['adaptive_settings']['min_target_sparsity']:.1f} - {results['adaptive_settings']['max_target_sparsity']:.1f}")
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise
