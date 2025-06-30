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
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AdaptiveArchitectureNet(nn.Module):
    def __init__(self, input_size, output_size=1, data_size=1000, complexity_factor=1.0):
        super(AdaptiveArchitectureNet, self).__init__()
        
        self.input_size = input_size
        
        if data_size < 500:
            hidden_size = max(32, input_size * 2)
            num_layers = 3
            dropout_rate = 0.1
        elif data_size < 2000:
            hidden_size = max(64, input_size * 3)
            num_layers = 4
            dropout_rate = 0.15
        elif data_size < 5000:
            hidden_size = max(128, input_size * 4)
            num_layers = 4
            dropout_rate = 0.2
        else:
            hidden_size = max(256, input_size * 4)
            num_layers = 5
            dropout_rate = 0.25
        
        hidden_size = int(hidden_size * complexity_factor)
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers - 1):
            next_size = hidden_size // (2 ** i) if i > 0 else hidden_size
            next_size = max(next_size, 32)
            
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

class ContinuousDQNAgent(nn.Module):
    def __init__(self, state_size=12, hidden_size=128):
        super(ContinuousDQNAgent, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        
        self.value = nn.Linear(hidden_size // 2, 1)
        self.pruning_ratio = nn.Linear(hidden_size // 2, 1)
        self.should_prune = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        value = self.value(x)
        pruning_ratio = torch.sigmoid(self.pruning_ratio(x))
        should_prune = torch.sigmoid(self.should_prune(x))
        
        return value, pruning_ratio, should_prune

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
    
    def get_performance_thresholds(self, historical_rmse=None, historical_mae=None):
        if historical_rmse is not None and len(historical_rmse) > 10:
            rmse_percentiles = np.percentile(historical_rmse, [10, 25, 50, 75, 90])
            mae_percentiles = np.percentile(historical_mae, [10, 25, 50, 75, 90]) if historical_mae else rmse_percentiles
            return {
                'rmse': {
                    'excellent': rmse_percentiles[0],
                    'good': rmse_percentiles[1],
                    'average': rmse_percentiles[2],
                    'poor': rmse_percentiles[3],
                    'bad': rmse_percentiles[4]
                },
                'mae': {
                    'excellent': mae_percentiles[0],
                    'good': mae_percentiles[1],
                    'average': mae_percentiles[2],
                    'poor': mae_percentiles[3],
                    'bad': mae_percentiles[4]
                }
            }
        else:
            scale_factor = self.stats['std'] if self.stats['std'] > 0 else 1.0
            return {
                'rmse': {
                    'excellent': scale_factor * 0.5,
                    'good': scale_factor * 1.0,
                    'average': scale_factor * 2.0,
                    'poor': scale_factor * 3.0,
                    'bad': scale_factor * 5.0
                },
                'mae': {
                    'excellent': scale_factor * 0.4,
                    'good': scale_factor * 0.8,
                    'average': scale_factor * 1.6,
                    'poor': scale_factor * 2.4,
                    'bad': scale_factor * 4.0
                }
            }
    
    def suggest_target_sparsity(self, data_size, complexity):
        base_sparsity = 0.3
        
        if data_size < 1000:
            size_factor = 0.7
        elif data_size < 5000:
            size_factor = 0.9
        else:
            size_factor = 1.1
        
        if complexity < 0.3:
            complexity_factor = 1.3
        elif complexity < 0.7:
            complexity_factor = 1.0
        else:
            complexity_factor = 0.8
        
        suggested_sparsity = base_sparsity * size_factor * complexity_factor
        return np.clip(suggested_sparsity, 0.1, 0.7)

class AdaptiveFedPruningRL:
    def __init__(self, input_size, num_clients=5, lr=0.001, all_y_data=None, data_sizes=None):
        self.num_clients = num_clients
        self.input_size = input_size
        self.lr = lr
        
        self.analyzer = DatasetAnalyzer()
        if all_y_data is not None:
            self.dataset_stats = self.analyzer.analyze(all_y_data)
            avg_data_size = np.mean(data_sizes) if data_sizes else 1000
            complexity = self.dataset_stats['cv']
            self.target_sparsity = self.analyzer.suggest_target_sparsity(avg_data_size, complexity)
        else:
            self.target_sparsity = 0.3
            self.dataset_stats = {}
        
        avg_data_size = np.mean(data_sizes) if data_sizes else 1000
        self.global_model = AdaptiveArchitectureNet(input_size, data_size=avg_data_size).to(device)
        self.client_models = [AdaptiveArchitectureNet(input_size, data_size=size).to(device) 
                             for size in (data_sizes or [avg_data_size] * num_clients)]
        self.client_masks = [None for _ in range(num_clients)]
        self.client_weights = [1.0 / num_clients for _ in range(num_clients)]
        
        self.original_model_size = self.global_model.get_model_size_mb()
        self.original_param_count = self.global_model.count_parameters()
        
        self.pruning_agent = ContinuousDQNAgent(state_size=12).to(device)
        self.target_agent = copy.deepcopy(self.pruning_agent)
        self.agent_optimizer = optim.Adam(self.pruning_agent.parameters(), lr=0.0003)
        
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.update_target_freq = 20
        self.round_count = 0
        
        self.best_global_rmse = float('inf')
        self.best_global_mae = float('inf')
        self.client_performance_history = {
            'rmse': [[] for _ in range(num_clients)],
            'mae': [[] for _ in range(num_clients)]
        }
        self.global_performance_history = {
            'rmse': [],
            'mae': [],
            'model_size_reduction': []
        }
        self.performance_thresholds = None
        
    def update_performance_thresholds(self):
        all_rmse = []
        all_mae = []
        for client_history in self.client_performance_history['rmse']:
            all_rmse.extend(client_history)
        for client_history in self.client_performance_history['mae']:
            all_mae.extend(client_history)
        
        if len(all_rmse) > 20:
            self.performance_thresholds = self.analyzer.get_performance_thresholds(all_rmse, all_mae)
        else:
            self.performance_thresholds = self.analyzer.get_performance_thresholds()
    
    def calculate_model_size_reduction(self):
        with torch.no_grad():
            total_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
            zero_params = sum((torch.abs(p) < 1e-6).sum().item() for p in self.global_model.parameters() 
                             if p.requires_grad and p.dtype.is_floating_point)
            
            active_params = total_params - zero_params
            size_reduction_percent = (1 - active_params / self.original_param_count) * 100
            return size_reduction_percent
        
    def get_adaptive_state(self, model, rmse, mae, loss, sparsity, round_num, client_id, data_size):
        with torch.no_grad():
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            zero_params = sum((torch.abs(p) < 1e-6).sum().item() for p in model.parameters() 
                             if p.requires_grad and p.dtype.is_floating_point)
            current_sparsity = zero_params / total_params if total_params > 0 else 0
            
            if self.performance_thresholds:
                rmse_percentile = 0.5
                for threshold_name, threshold_val in self.performance_thresholds['rmse'].items():
                    if rmse <= threshold_val:
                        rmse_percentile = {'excellent': 0.9, 'good': 0.7, 'average': 0.5, 
                                         'poor': 0.3, 'bad': 0.1}.get(threshold_name, 0.5)
                        break
                        
                mae_percentile = 0.5
                for threshold_name, threshold_val in self.performance_thresholds['mae'].items():
                    if mae <= threshold_val:
                        mae_percentile = {'excellent': 0.9, 'good': 0.7, 'average': 0.5, 
                                        'poor': 0.3, 'bad': 0.1}.get(threshold_name, 0.5)
                        break
            else:
                rmse_percentile = np.clip(1.0 - (rmse / (self.dataset_stats.get('std', 1.0) * 3)), 0, 1)
                mae_percentile = np.clip(1.0 - (mae / (self.dataset_stats.get('std', 1.0) * 2.5)), 0, 1)
            
            normalized_loss = np.clip(loss / (self.dataset_stats.get('std', 1.0) ** 2), 0, 2)
            normalized_round = min(round_num / 100.0, 1.0)
            normalized_client = client_id / max(1, self.num_clients - 1)
            normalized_data_size = np.clip(data_size / 5000.0, 0, 1)
            
            improvement = 0.0
            if len(self.client_performance_history['rmse'][client_id]) > 1:
                recent_history = self.client_performance_history['rmse'][client_id][-5:]
                if len(recent_history) >= 2:
                    improvement = np.clip((recent_history[0] - recent_history[-1]) / max(recent_history[0], 1e-6), -1, 1)
            
            global_improvement = 0.0
            if len(self.global_performance_history['rmse']) > 1:
                recent_global = self.global_performance_history['rmse'][-5:]
                if len(recent_global) >= 2:
                    global_improvement = np.clip((recent_global[0] - recent_global[-1]) / max(recent_global[0], 1e-6), -1, 1)
            
            sparsity_gap = np.clip(self.target_sparsity - current_sparsity, -1, 1)
            
            model_complexity = total_params / (self.input_size * 1000)
            model_complexity = np.clip(model_complexity, 0, 2)
            
            training_stability = 0.0
            if len(self.client_performance_history['rmse'][client_id]) > 3:
                recent_rmse = self.client_performance_history['rmse'][client_id][-3:]
                if len(recent_rmse) >= 3:
                    stability_var = np.var(recent_rmse)
                    training_stability = np.clip(1.0 / (1.0 + stability_var), 0, 1)
            
            combined_performance = (rmse_percentile + mae_percentile) / 2
            
            state = torch.tensor([
                combined_performance,
                1.0 - normalized_loss,
                current_sparsity,
                normalized_round,
                normalized_client,
                normalized_data_size,
                improvement,
                global_improvement,
                sparsity_gap,
                model_complexity,
                training_stability,
                np.clip(self.target_sparsity, 0, 1)
            ], dtype=torch.float32).to(device)
            
            return state
    
    def get_continuous_action(self, state):
        if np.random.random() <= self.epsilon:
            value = torch.tensor([0.0]).to(device)
            pruning_ratio = torch.tensor([np.random.uniform(0, 0.5)]).to(device)
            should_prune = torch.tensor([np.random.uniform(0, 1)]).to(device)
            return value, pruning_ratio, should_prune
        
        with torch.no_grad():
            value, pruning_ratio, should_prune = self.pruning_agent(state.unsqueeze(0))
            return value.squeeze(), pruning_ratio.squeeze(), should_prune.squeeze()
    
    def remember_continuous(self, state, action_tuple, reward, next_state, done):
        self.memory.append((state, action_tuple, reward, next_state, done))
    
    def replay_continuous(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([e[0] for e in batch]).to(device)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(device)
        next_states = torch.stack([e[3] for e in batch]).to(device)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.float32).to(device)
        
        current_values, current_pruning, current_should_prune = self.pruning_agent(states)
        
        with torch.no_grad():
            next_values, _, _ = self.target_agent(next_states)
            target_values = rewards + (self.gamma * next_values.squeeze() * (1 - dones))
        
        value_loss = F.mse_loss(current_values.squeeze(), target_values)
        
        pruning_targets = torch.zeros_like(current_pruning.squeeze())
        prune_targets = torch.zeros_like(current_should_prune.squeeze())
        
        for i, (state, action_tuple, reward, next_state, done) in enumerate(batch):
            if reward > 0:
                pruning_targets[i] = action_tuple[1]
                prune_targets[i] = action_tuple[2]
        
        pruning_loss = F.mse_loss(current_pruning.squeeze(), pruning_targets)
        prune_decision_loss = F.binary_cross_entropy(current_should_prune.squeeze(), prune_targets)
        
        total_loss = value_loss + 0.5 * pruning_loss + 0.3 * prune_decision_loss
        
        self.agent_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pruning_agent.parameters(), 1.0)
        self.agent_optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def adaptive_structured_prune(self, model, pruning_ratio, should_prune_prob, client_id):
        if should_prune_prob < 0.5 or pruning_ratio < 0.01:
            return {name: torch.ones_like(param) for name, param in model.named_parameters() 
                   if param.requires_grad and param.dtype.is_floating_point}
        
        pruning_ratio = min(pruning_ratio, 0.6)
        
        mask = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype.is_floating_point:
                if len(param.shape) > 1:
                    flat_weights = param.data.abs().flatten()
                    if len(flat_weights) > 0:
                        threshold_idx = int(len(flat_weights) * pruning_ratio)
                        if threshold_idx < len(flat_weights):
                            threshold = torch.kthvalue(flat_weights, threshold_idx + 1).values
                            param_mask = (param.data.abs() > threshold).float()
                        else:
                            param_mask = torch.zeros_like(param).float()
                        
                        min_keep_ratio = max(0.05, 1.0 - self.target_sparsity * 1.5)
                        if param_mask.sum() < param.numel() * min_keep_ratio:
                            sorted_weights = param.data.abs().flatten().sort(descending=True)
                            keep_count = max(int(param.numel() * min_keep_ratio), 1)
                            if keep_count < len(sorted_weights.values):
                                threshold_val = sorted_weights.values[keep_count-1]
                                param_mask = (param.data.abs() >= threshold_val).float()
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
    
    def aggregate_models(self, client_models, client_masks, client_data_sizes):
        global_dict = self.global_model.state_dict()
        total_data_size = sum(client_data_sizes)
        
        if total_data_size == 0:
            return
        
        weights = [size / total_data_size for size in client_data_sizes]
        
        for key in global_dict.keys():
            if any(mask is not None and key in mask for mask in client_masks):
                overlaps = torch.zeros_like(global_dict[key])
                weighted_sum = torch.zeros_like(global_dict[key])
                
                for i, (model, mask, weight) in enumerate(zip(client_models, client_masks, weights)):
                    if mask is not None and key in mask:
                        client_dict = model.state_dict()
                        client_weights = client_dict[key] * mask[key]
                        overlaps += mask[key] * weight
                        weighted_sum += client_weights * weight
                
                overlaps = torch.clamp(overlaps, min=1e-8)
                global_dict[key] = weighted_sum / overlaps
            else:
                weighted_params = []
                for i, model in enumerate(client_models):
                    param = model.state_dict()[key]
                    if param.dtype in [torch.long, torch.int64, torch.int32]:
                        weighted_params.append(param.float() * weights[i])
                    else:
                        weighted_params.append(param * weights[i])
                
                if len(weighted_params) > 0:
                    summed = torch.stack(weighted_params).sum(0)
                    if global_dict[key].dtype in [torch.long, torch.int64, torch.int32]:
                        global_dict[key] = summed.long()
                    else:
                        global_dict[key] = summed
        
        self.global_model.load_state_dict(global_dict)
    
    def calculate_adaptive_reward(self, prev_rmse, curr_rmse, prev_mae, curr_mae, prev_sparsity, curr_sparsity, client_id):
        if self.performance_thresholds is None:
            self.update_performance_thresholds()
        
        rmse_improvement = (prev_rmse - curr_rmse) / max(prev_rmse, 1e-6)
        mae_improvement = (prev_mae - curr_mae) / max(prev_mae, 1e-6)
        
        combined_improvement = (rmse_improvement + mae_improvement) / 2
        
        rmse_reward = 0
        mae_reward = 0
        
        if self.performance_thresholds:
            if curr_rmse <= self.performance_thresholds['rmse']['excellent']:
                rmse_reward = rmse_improvement * 25 + 20
            elif curr_rmse <= self.performance_thresholds['rmse']['good']:
                rmse_reward = rmse_improvement * 17.5 + 12.5
            elif curr_rmse <= self.performance_thresholds['rmse']['average']:
                rmse_reward = rmse_improvement * 12.5 + 7.5
            elif curr_rmse <= self.performance_thresholds['rmse']['poor']:
                rmse_reward = rmse_improvement * 7.5 + 2.5
            else:
                rmse_reward = rmse_improvement * 5 - 2.5
                
            if curr_mae <= self.performance_thresholds['mae']['excellent']:
                mae_reward = mae_improvement * 25 + 20
            elif curr_mae <= self.performance_thresholds['mae']['good']:
                mae_reward = mae_improvement * 17.5 + 12.5
            elif curr_mae <= self.performance_thresholds['mae']['average']:
                mae_reward = mae_improvement * 12.5 + 7.5
            elif curr_mae <= self.performance_thresholds['mae']['poor']:
                mae_reward = mae_improvement * 7.5 + 2.5
            else:
                mae_reward = mae_improvement * 5 - 2.5
        else:
            scale = self.dataset_stats.get('std', 1.0)
            if curr_rmse <= scale * 0.5:
                rmse_reward = rmse_improvement * 25 + 20
            elif curr_rmse <= scale * 1.0:
                rmse_reward = rmse_improvement * 17.5 + 12.5
            elif curr_rmse <= scale * 2.0:
                rmse_reward = rmse_improvement * 12.5 + 7.5
            else:
                rmse_reward = rmse_improvement * 5
                
            if curr_mae <= scale * 0.4:
                mae_reward = mae_improvement * 25 + 20
            elif curr_mae <= scale * 0.8:
                mae_reward = mae_improvement * 17.5 + 12.5
            elif curr_mae <= scale * 1.6:
                mae_reward = mae_improvement * 12.5 + 7.5
            else:
                mae_reward = mae_improvement * 5
        
        sparsity_diff = abs(curr_sparsity - self.target_sparsity)
        if sparsity_diff < 0.02:
            sparsity_reward = 20
        elif sparsity_diff < 0.05:
            sparsity_reward = 15
        elif sparsity_diff < 0.1:
            sparsity_reward = 10
        elif sparsity_diff < 0.2:
            sparsity_reward = 5
        else:
            sparsity_reward = -5
        
        efficiency_bonus = 0
        if curr_sparsity > self.target_sparsity * 0.8 and curr_rmse <= prev_rmse + self.dataset_stats.get('std', 1.0) * 0.1:
            efficiency_bonus = 20
        elif curr_sparsity > self.target_sparsity * 0.6 and curr_rmse <= prev_rmse + self.dataset_stats.get('std', 1.0) * 0.2:
            efficiency_bonus = 10
        
        stability_penalty = 0
        threshold_degradation = self.dataset_stats.get('std', 1.0) * 0.5
        if curr_rmse > prev_rmse + threshold_degradation * 2:
            stability_penalty = -30
        elif curr_rmse > prev_rmse + threshold_degradation:
            stability_penalty = -15
        
        performance_reward = (rmse_reward + mae_reward) / 2
        total_reward = performance_reward + sparsity_reward + efficiency_bonus + stability_penalty
        return np.clip(total_reward, -50, 100)
    
    def train_client(self, client_id, train_loader, val_loader, epochs=20, round_num=0):
        model = self.client_models[client_id]
        model.load_state_dict(self.global_model.state_dict())
        
        data_size = len(train_loader.dataset)
        
        adaptive_lr = self.lr
        if round_num > 80:
            adaptive_lr *= 0.05
        elif round_num > 60:
            adaptive_lr *= 0.1
        elif round_num > 40:
            adaptive_lr *= 0.3
        elif round_num > 20:
            adaptive_lr *= 0.7
        
        optimizer = optim.AdamW(model.parameters(), lr=adaptive_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()
        
        prev_rmse = self.client_performance_history['rmse'][client_id][-1] if self.client_performance_history['rmse'][client_id] else self.dataset_stats.get('std', 10.0)
        prev_mae = self.client_performance_history['mae'][client_id][-1] if self.client_performance_history['mae'][client_id] else self.dataset_stats.get('std', 8.0)
        prev_sparsity = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                
                if self.client_masks[client_id] is not None:
                    self.apply_mask(model, self.client_masks[client_id])
                
                epoch_loss += loss.item()
                num_batches += 1
            
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
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        zero_params = sum((torch.abs(p) < 1e-6).sum().item() for p in model.parameters() 
                         if p.requires_grad and p.dtype.is_floating_point)
        current_sparsity = zero_params / total_params if total_params > 0 else 0
        
        state = self.get_adaptive_state(model, rmse, mae, avg_loss, current_sparsity, 
                                       round_num, client_id, data_size)
        value, pruning_ratio, should_prune = self.get_continuous_action(state)
        
        action_tuple = (value.item(), pruning_ratio.item(), should_prune.item())
        
        if should_prune.item() > 0.5 and current_sparsity < 0.8 and round_num >= 5:
            mask = self.adaptive_structured_prune(model, pruning_ratio.item(), should_prune.item(), client_id)
            self.apply_mask(model, mask)
            self.client_masks[client_id] = mask
            
            model.train()
            fine_tune_optimizer = optim.AdamW(model.parameters(), lr=adaptive_lr*0.1, 
                                            weight_decay=1e-4)
            for ft_epoch in range(3):
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    fine_tune_optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    fine_tune_optimizer.step()
                    self.apply_mask(model, mask)
        
        new_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        new_zero_params = sum((torch.abs(p) < 1e-6).sum().item() for p in model.parameters() 
                             if p.requires_grad and p.dtype.is_floating_point)
        new_sparsity = new_zero_params / new_total_params if new_total_params > 0 else 0
        
        model.eval()
        new_val_predictions = []
        new_val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                new_val_predictions.extend(output.cpu().numpy().flatten())
                new_val_targets.extend(target.cpu().numpy().flatten())
        
        new_rmse = np.sqrt(mean_squared_error(new_val_targets, new_val_predictions))
        new_mae = mean_absolute_error(new_val_targets, new_val_predictions)
        
        self.client_performance_history['rmse'][client_id].append(new_rmse)
        self.client_performance_history['mae'][client_id].append(new_mae)
        
        next_state = self.get_adaptive_state(model, new_rmse, new_mae, avg_loss, new_sparsity, 
                                           round_num, client_id, data_size)
        reward = self.calculate_adaptive_reward(prev_rmse, new_rmse, prev_mae, new_mae, prev_sparsity, new_sparsity, client_id)
        
        self.remember_continuous(state, action_tuple, reward, next_state, False)
        if len(self.memory) > 128:
            self.replay_continuous(batch_size=64)
        
        return new_rmse, new_mae, avg_loss, new_sparsity, data_size
    
    def federated_round(self, client_loaders, val_loaders, round_num):
        rmses = []
        maes = []
        losses = []
        sparsities = []
        data_sizes = []
        
        active_clients = min(len(client_loaders), self.num_clients)
        participation_rate = max(0.3, 1.0 - round_num * 0.005)
        num_selected = max(1, int(active_clients * participation_rate))
        selected_clients = random.sample(range(len(client_loaders)), num_selected)
        
        for client_id in selected_clients:
            if len(client_loaders) > client_id and len(val_loaders) > client_id:
                rmse, mae, loss, sparsity, data_size = self.train_client(
                    client_id, 
                    client_loaders[client_id], 
                    val_loaders[client_id],
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
            self.aggregate_models(active_models, active_masks, data_sizes)
            
            avg_rmse = np.mean(rmses)
            avg_mae = np.mean(maes)
            model_size_reduction = self.calculate_model_size_reduction()
            
            self.global_performance_history['rmse'].append(avg_rmse)
            self.global_performance_history['mae'].append(avg_mae)
            self.global_performance_history['model_size_reduction'].append(model_size_reduction)
            
            if round_num % 10 == 0:
                self.update_performance_thresholds()
        
        self.round_count += 1
        if self.round_count % self.update_target_freq == 0:
            self.target_agent.load_state_dict(self.pruning_agent.state_dict())
        
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

def run_adaptive_experiment():
    
    df, loc_df = load_real_soybean_data()
    
    X, y, metadata, feature_cols = preprocess_real_data(df, loc_df)
    
    scaler = RobustScaler()
    X_normalized = scaler.fit_transform(X)
    
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=None
    )
    
    train_metadata = metadata[:len(X_train_all)]
    
    num_clients = min(8, max(3, len(X_train_all) // 200))
    client_data, client_sizes = create_federated_clients_real(X_train_all, y_train_all, train_metadata, 
                                                              num_clients=num_clients)
    
    if len(client_data) == 0:
        raise ValueError("No valid client data created. Check data quality and size.")
    
    input_size = X_normalized.shape[1]
    fed_system = AdaptiveFedPruningRL(
        input_size=input_size,
        num_clients=len(client_data),
        lr=0.001,
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
    
    num_rounds = 120
    results = {
        'train_rmse': [],
        'val_rmse': [],
        'test_rmse': [],
        'test_mae': [],
        'loss': [],
        'sparsity': [],
        'epsilon': [],
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
        
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        
        communication_cost = (1 - avg_sparsity) * 100
        model_size_reduction = fed_system.calculate_model_size_reduction()
        
        results['train_rmse'].append(avg_train_rmse)
        results['val_rmse'].append(avg_train_rmse)
        results['test_rmse'].append(test_rmse)
        results['test_mae'].append(test_mae)
        results['loss'].append(avg_loss)
        results['sparsity'].append(avg_sparsity)
        results['epsilon'].append(fed_system.epsilon)
        results['communication_cost'].append(communication_cost)
        results['target_sparsity'].append(fed_system.target_sparsity)
        results['model_size_reduction'].append(model_size_reduction)
        
        if test_rmse < best_test_rmse or test_mae < best_test_mae:
            best_test_rmse = min(best_test_rmse, test_rmse)
            best_test_mae = min(best_test_mae, test_mae)
            best_round = round_num
            patience_counter = 0
            
            torch.save({
                'model_state_dict': fed_system.global_model.state_dict(),
                'round': round_num,
                'rmse': test_rmse,
                'mae': test_mae,
                'sparsity': avg_sparsity,
                'target_sparsity': fed_system.target_sparsity,
                'model_size_reduction': model_size_reduction
            }, 'best_adaptive_model.pth')
        else:
            patience_counter += 1
        
        if round_num % 5 == 0 or round_num == num_rounds - 1:
            print(f"Round {round_num + 1:3d}: "
                  f"RMSE: {test_rmse:.4f} | "
                  f"MAE: {test_mae:.4f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Sparsity: {avg_sparsity:.3f} | "
                  f"Model Size Reduction: {model_size_reduction:.1f}% | "
                  f"R²: {test_r2:.4f} | "
                  f"ε: {fed_system.epsilon:.3f}")
        
        if patience_counter >= patience_limit:
            break
    
    try:
        checkpoint = torch.load('best_adaptive_model.pth')
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
    
    final_rmse = np.sqrt(mean_squared_error(final_targets, final_predictions))
    final_mae = mean_absolute_error(final_targets, final_predictions)
    final_r2 = r2_score(final_targets, final_predictions)
    final_mape = np.mean(np.abs((np.array(final_targets) - np.array(final_predictions)) / 
                               np.maximum(np.array(final_targets), 1e-6))) * 100
    
    total_params = sum(p.numel() for p in fed_system.global_model.parameters() if p.requires_grad)
    zero_params = sum((torch.abs(p) < 1e-6).sum().item() for p in fed_system.global_model.parameters() 
                     if p.requires_grad and p.dtype.is_floating_point)
    final_sparsity = zero_params / total_params if total_params > 0 else 0
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
    plt.plot(results['sparsity'], color='purple', linewidth=2, label='Actual')
    plt.plot(results['target_sparsity'], color='orange', linewidth=2, linestyle='--', label='Target')
    plt.xlabel('Round')
    plt.ylabel('Sparsity')
    plt.title('Model Sparsity Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 4)
    plt.plot(results['model_size_reduction'], color='red', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Model Size Reduction (%)')
    plt.title('Model Size Reduction')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 5)
    plt.plot(results['loss'], color='brown', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 6)
    plt.plot(results['epsilon'], color='gray', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.title('RL Exploration Decay')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 7)
    plt.scatter(results['sparsity'], results['test_rmse'], alpha=0.6, color='red', s=30, label='RMSE')
    plt.scatter(results['sparsity'], results['test_mae'], alpha=0.6, color='green', s=30, label='MAE')
    plt.xlabel('Sparsity')
    plt.ylabel('Error')
    plt.title('Efficiency vs Performance Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 8)
    plt.plot(results['communication_cost'], color='cyan', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Communication Cost (%)')
    plt.title('Communication Efficiency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 9)
    plt.scatter(final_targets, final_predictions, alpha=0.6, s=20)
    min_val, max_val = min(min(final_targets), min(final_predictions)), max(max(final_targets), max(final_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'Predictions vs Actual (R² = {final_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 10)
    residuals = np.array(final_targets) - np.array(final_predictions)
    plt.scatter(final_predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Yield')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 11)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 3, 12)
    plt.plot(results['test_rmse'], label='RMSE', color='red', alpha=0.8)
    plt.plot(results['test_mae'], label='MAE', color='green', alpha=0.8)
    plt.xlabel('Round')
    plt.ylabel('Error Metrics')
    plt.title('RMSE vs MAE Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_adaptive_soybean_rl_federated_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'final_rmse': final_rmse,
        'final_mae': final_mae,
        'final_r2': final_r2,
        'final_mape': final_mape,
        'final_sparsity': final_sparsity,
        'target_sparsity': fed_system.target_sparsity,
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
        'original_param_count': fed_system.original_param_count
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    try:
        results = run_adaptive_experiment()
        
        print(f"Final RMSE: {results['final_rmse']:.4f}")
        print(f"Final MAE: {results['final_mae']:.4f}")
        print(f"Final R²: {results['final_r2']:.4f}")
        print(f"Best RMSE: {results['best_rmse']:.4f}")
        print(f"Best MAE: {results['best_mae']:.4f}")
        print(f"Model Compression: {results['compression_ratio']:.1f}x")
        print(f"Communication Savings: {results['communication_savings']:.1f}%")
        print(f"Model Size Reduction: {results['model_size_reduction']:.1f}%")
        print(f"Target Sparsity: {results['target_sparsity']:.3f}")
        print(f"Achieved Sparsity: {results['final_sparsity']:.3f}")
        print(f"Original Model Size: {results['original_model_size_mb']:.2f} MB")
        print(f"Original Parameters: {results['original_param_count']:,}")
        print(f"Clients: {results['num_clients']}")
        print(f"Features: {results['input_features']}")
        print(f"Train Samples: {results['train_samples']}")
        print(f"Test Samples: {results['test_samples']}")
        print(f"Total Rounds: {results['total_rounds']}")
        
    except Exception as e:
        print(f"Experiment failed: {str(e)}")
        raise
