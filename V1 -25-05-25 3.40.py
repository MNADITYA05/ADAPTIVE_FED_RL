import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import scipy.stats as stats
from scipy.signal import find_peaks
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

class EnhancedCLASModelWithAnswers:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_names = []
        self.processed_data = []
        
    def load_participant_answers(self, participant_id):
        """Load task performance data from Answers directory"""
        answers_file = os.path.join(self.data_path, f"Answers/Part{participant_id}_c_i_answers.csv")
        
        if os.path.exists(answers_file):
            try:
                answers_df = pd.read_csv(answers_file)
                print(f"  Loaded answers for Participant {participant_id}: {answers_df.shape}")
                print(f"  Columns: {list(answers_df.columns)}")
                return answers_df
            except Exception as e:
                print(f"  Error loading answers for Participant {participant_id}: {e}")
                return None
        else:
            print(f"  No answers file found for Participant {participant_id}")
            return None
    
    def load_participant_block_details(self, participant_id):
        """Load block details to map signals to tasks"""
        block_details_file = os.path.join(self.data_path, f"Block_details/Part{participant_id}_Block_Details.csv")
        
        if os.path.exists(block_details_file):
            try:
                block_df = pd.read_csv(block_details_file)
                print(f"  Loaded block details for Participant {participant_id}: {block_df.shape}")
                return block_df
            except Exception as e:
                print(f"  Error loading block details for Participant {participant_id}: {e}")
                return None
        return None
    
    def extract_task_performance_features(self, answers_df, block_details_df):
        """Extract task performance features from answers data"""
        if answers_df is None:
            return {}
        
        features = {}
        print(f"    Available columns: {list(answers_df.columns)}")
        
        # Check what columns we actually have
        answer_col = None
        stimulus_col = None
        
        # Find answer column (could be 'answer', ' answer', 'Answer', etc.)
        for col in answers_df.columns:
            if 'answer' in col.lower().strip():
                answer_col = col
                break
        
        # Find stimulus column
        for col in answers_df.columns:
            if 'stimulus' in col.lower().strip():
                stimulus_col = col
                break
        
        if answer_col is None:
            print(f"    No answer column found")
            return {}
        
        print(f"    Using answer column: '{answer_col}'")
        print(f"    Sample answers: {answers_df[answer_col].head().tolist()}")
        
        # Analyze answer patterns to determine correctness
        answers = answers_df[answer_col].astype(str).str.strip()
        
        # Try to determine correct/incorrect based on answer patterns
        # Common patterns: 'correct'/'incorrect', '1'/'0', 'true'/'false', etc.
        unique_answers = answers.unique()
        print(f"    Unique answers: {unique_answers}")
        
        # Create correctness mapping
        correctness_values = []
        for answer in answers:
            answer_lower = answer.lower()
            if answer_lower in ['correct', '1', 'true', 'yes', 'right']:
                correctness_values.append(1)
            elif answer_lower in ['incorrect', '0', 'false', 'no', 'wrong']:
                correctness_values.append(0)
            else:
                # Try to parse as number - assume higher numbers = better performance
                try:
                    val = float(answer)
                    # If it's a score, normalize to 0-1 range
                    if val >= 0:
                        correctness_values.append(min(val, 1.0))
                    else:
                        correctness_values.append(0.5)  # Default
                except:
                    correctness_values.append(0.5)  # Default neutral
        
        if len(correctness_values) > 0:
            # Basic performance metrics
            features['task_accuracy'] = np.mean(correctness_values)
            features['task_error_rate'] = 1 - features['task_accuracy']
            features['total_responses'] = len(correctness_values)
            features['correct_responses'] = sum(correctness_values)
            features['incorrect_responses'] = len(correctness_values) - sum(correctness_values)
        
        # Analyze stimulus patterns for task-specific performance
        if stimulus_col is not None:
            stimuli = answers_df[stimulus_col].astype(str)
            
            # Group by stimulus types
            stimulus_performance = {}
            for i, stimulus in enumerate(stimuli):
                if i < len(correctness_values):
                    stimulus_lower = stimulus.lower()
                    
                    # Categorize stimulus type
                    if any(word in stimulus_lower for word in ['math', 'arithmetic', '+', '-', '*', '/']):
                        task_type = 'math'
                    elif any(word in stimulus_lower for word in ['logic', 'reasoning', 'puzzle']):
                        task_type = 'logic'
                    elif any(word in stimulus_lower for word in ['stroop', 'color', 'word']):
                        task_type = 'stroop'
                    else:
                        task_type = 'general'
                    
                    if task_type not in stimulus_performance:
                        stimulus_performance[task_type] = []
                    stimulus_performance[task_type].append(correctness_values[i])
            
            # Calculate task-specific accuracies
            for task_type, performances in stimulus_performance.items():
                features[f'{task_type}_accuracy'] = np.mean(performances)
                features[f'{task_type}_consistency'] = 1 - np.std(performances) if len(performances) > 1 else 1
        
        # Performance decline analysis
        if len(correctness_values) > 10:
            mid_point = len(correctness_values) // 2
            first_half_acc = np.mean(correctness_values[:mid_point])
            second_half_acc = np.mean(correctness_values[mid_point:])
            
            features['performance_decline'] = first_half_acc - second_half_acc
            features['fatigue_indicator'] = 1 if features['performance_decline'] > 0.1 else 0
            
            # Consistency measures
            features['performance_variability'] = np.std(correctness_values)
            features['performance_trend'] = np.corrcoef(range(len(correctness_values)), correctness_values)[0,1]
        
        # Response pattern analysis
        if answer_col:
            # Count response types
            response_counts = answers_df[answer_col].value_counts()
            features['response_diversity'] = len(response_counts)
            features['most_common_response_freq'] = response_counts.iloc[0] / len(answers_df) if len(response_counts) > 0 else 0
        
        print(f"    Extracted {len(features)} task performance features")
        
        return features
    
    def enhanced_stimulus_categorization(self, stimulus_type, task_performance_features):
        """Enhanced stimulus categorization using task performance"""
        base_stress, base_workload = self.create_labels(stimulus_type)
        
        # Adjust labels based on task performance
        if task_performance_features:
            # High error rate or slow responses indicate higher stress/workload
            error_rate = task_performance_features.get('task_error_rate', 0)
            avg_response_time = task_performance_features.get('avg_response_time', 1000)
            
            # Stress adjustment
            if error_rate > 0.3 or avg_response_time > 2000:  # High errors or very slow
                adjusted_stress = min(base_stress + 1, 3)  # Cap at 3
            elif error_rate < 0.1 and avg_response_time < 1000:  # Low errors and fast
                adjusted_stress = max(base_stress - 1, 0)  # Floor at 0
            else:
                adjusted_stress = base_stress
                
            # Workload adjustment
            if error_rate > 0.4 or avg_response_time > 2500:
                adjusted_workload = min(base_workload + 1, 3)
            elif error_rate < 0.05 and avg_response_time < 800:
                adjusted_workload = max(base_workload - 1, 0)
            else:
                adjusted_workload = base_workload
                
            return adjusted_stress, adjusted_workload
        
        return base_stress, base_workload
    
    def process_participant_with_answers(self, participant_id, max_files_per_type=30):
        """Enhanced processing including task performance data"""
        print(f"\nProcessing Participant {participant_id} (Enhanced with Answers)...")
        
        # Load task performance data
        answers_df = self.load_participant_answers(participant_id)
        block_details_df = self.load_participant_block_details(participant_id)
        
        # Extract task performance features
        task_performance_features = self.extract_task_performance_features(answers_df, block_details_df)
        print(f"  Task performance features extracted: {len(task_performance_features)}")
        
        # Load stimulus metadata
        stimulus_info = self.load_stimulus_metadata(participant_id)
        
        # Process physiological signals
        participant_folder = os.path.join(self.data_path, f"Participants/Part{participant_id}/by_block")
        if not os.path.exists(participant_folder):
            participant_folder = os.path.join(self.data_path, f"Participants/Part{participant_id}/all_separate")
        
        ecg_files = glob.glob(os.path.join(participant_folder, "*_ecg_*.csv"))
        gsr_files = glob.glob(os.path.join(participant_folder, "*_gsr_ppg_*.csv"))
        
        # Sample files to manage processing
        if len(ecg_files) > max_files_per_type:
            ecg_files = np.random.choice(ecg_files, max_files_per_type, replace=False).tolist()
        if len(gsr_files) > max_files_per_type:
            gsr_files = np.random.choice(gsr_files, max_files_per_type, replace=False).tolist()
        
        processed_records = []
        
        # Process multimodal files (GSR/PPG/ACC) - these are more comprehensive
        for gsr_file in gsr_files:
            try:
                gsr_df = pd.read_csv(gsr_file)
                if len(gsr_df) > 50:
                    # Extract physiological signals
                    gsr_signal = gsr_df['gsr'].values if 'gsr' in gsr_df.columns else gsr_df.iloc[:, -1].values
                    ppg_signal = gsr_df['ppg'].values if 'ppg' in gsr_df.columns else gsr_df.iloc[:, -2].values
                    acc_data = gsr_df[['accelx', 'accely', 'accelz']].values if all(col in gsr_df.columns for col in ['accelx', 'accely', 'accelz']) else gsr_df.iloc[:, 2:5].values
                    
                    if np.var(gsr_signal) > 0:
                        stimulus_type = self.get_stimulus_type_from_metadata(gsr_file, stimulus_info)
                        
                        # Extract physiological features
                        gsr_features = self.extract_eda_features(gsr_signal)
                        ppg_features = self.extract_ppg_features(ppg_signal)
                        acc_features = self.extract_accelerometer_features(acc_data)
                        
                        # Combine physiological and task performance features
                        combined_features = {**gsr_features, **ppg_features, **acc_features, **task_performance_features}
                        
                        # Enhanced label creation using task performance
                        stress_level, workload_level = self.enhanced_stimulus_categorization(stimulus_type, task_performance_features)
                        
                        record = {
                            'participant_id': participant_id,
                            'stimulus_type': stimulus_type,
                            'features': combined_features,
                            'stress_level': stress_level,
                            'workload_level': workload_level,
                            'signal_type': 'multimodal_with_performance',
                            'file': os.path.basename(gsr_file),
                            'has_task_data': len(task_performance_features) > 0
                        }
                        processed_records.append(record)
                        
            except Exception as e:
                continue
        
        print(f"  Processed {len(processed_records)} enhanced records for Participant {participant_id}")
        if task_performance_features:
            task_acc = task_performance_features.get('task_accuracy', 0)
            if isinstance(task_acc, (int, float)):
                print(f"  Task accuracy: {task_acc:.3f}")
            else:
                print(f"  Task accuracy: {task_acc}")
            
            response_time = task_performance_features.get('avg_response_time', 0)
            if isinstance(response_time, (int, float)):
                print(f"  Avg response time: {response_time:.1f}ms")
            else:
                print(f"  Avg response time: {response_time}")
        
        return processed_records
    
    def load_stimulus_metadata(self, participant_id):
        """Load stimulus metadata"""
        meta_gsr_file = os.path.join(self.data_path, f"Participants/Part{participant_id}/meta_info_gsr_ppg.csv")
        
        stimulus_info = {}
        if os.path.exists(meta_gsr_file):
            stimulus_info['gsr_meta'] = pd.read_csv(meta_gsr_file)
            
        return stimulus_info
    
    def get_stimulus_type_from_metadata(self, filename, stimulus_info):
        """Get stimulus type from metadata or filename"""
        return self.extract_stimulus_from_filename(filename)
    
    def extract_stimulus_from_filename(self, filename):
        """Extract stimulus type from filename"""
        filename_lower = os.path.basename(filename).lower()
        
        import re
        numbers = re.findall(r'\d+', filename_lower)
        if numbers:
            try:
                stimulus_num = int(numbers[0]) % 100
                if stimulus_num <= 10:
                    return 'neutral'
                elif stimulus_num <= 30:
                    return 'math'
                elif stimulus_num <= 50:
                    return 'logic'
                elif stimulus_num <= 70:
                    return 'stroop'
                elif stimulus_num <= 85:
                    return 'pictures'
                else:
                    return 'videos'
            except:
                pass
        
        return 'neutral'
    
    def create_labels(self, stimulus_type):
        """Create basic stress and workload labels"""
        stress_mapping = {'neutral': 0, 'pictures': 1, 'videos': 1, 'math': 2, 'logic': 2, 'stroop': 3}
        workload_mapping = {'neutral': 0, 'pictures': 0, 'videos': 0, 'stroop': 2, 'math': 3, 'logic': 3}
        
        return stress_mapping.get(stimulus_type, 1), workload_mapping.get(stimulus_type, 1)
    
    # Include all the feature extraction methods from the previous version
    def extract_eda_features(self, eda_signal, fs=256):
        """Extract EDA/GSR features"""
        features = {}
        
        features['eda_mean'] = np.mean(eda_signal)
        features['eda_std'] = np.std(eda_signal)
        features['eda_range'] = np.max(eda_signal) - np.min(eda_signal)
        features['eda_slope'] = stats.linregress(range(len(eda_signal)), eda_signal)[0]
        
        try:
            eda_smooth = np.convolve(eda_signal, np.ones(int(fs*0.5))/int(fs*0.5), mode='same')
            eda_deriv = np.diff(eda_smooth)
            
            threshold = np.std(eda_deriv) * 2
            scr_peaks, _ = find_peaks(eda_deriv, height=threshold, distance=int(fs*1))
            
            features['scr_count'] = len(scr_peaks)
            features['scr_rate'] = len(scr_peaks) / (len(eda_signal) / fs)
            
            if len(scr_peaks) > 0:
                features['scr_amplitude_mean'] = np.mean(eda_deriv[scr_peaks])
            else:
                features['scr_amplitude_mean'] = 0
        except:
            features['scr_count'] = 0
            features['scr_rate'] = 0
            features['scr_amplitude_mean'] = 0
        
        return features
    
    def extract_ppg_features(self, ppg_signal, fs=256):
        """Extract PPG features"""
        features = {}
        
        features['ppg_mean'] = np.mean(ppg_signal)
        features['ppg_std'] = np.std(ppg_signal)
        features['ppg_range'] = np.max(ppg_signal) - np.min(ppg_signal)
        features['ppg_energy'] = np.sum(ppg_signal**2)
        
        try:
            peaks, _ = find_peaks(ppg_signal, height=np.mean(ppg_signal))
            if len(peaks) > 2:
                features['ppg_peak_rate'] = len(peaks) / (len(ppg_signal) / fs)
                features['ppg_hr_estimate'] = 60 / np.mean(np.diff(peaks) / fs)
            else:
                features['ppg_peak_rate'] = 0
                features['ppg_hr_estimate'] = 70
        except:
            features['ppg_peak_rate'] = 0
            features['ppg_hr_estimate'] = 70
        
        return features
    
    def extract_accelerometer_features(self, acc_data, fs=256):
        """Extract accelerometer features"""
        features = {}
        
        if acc_data.ndim == 1:
            acc_data = acc_data.reshape(-1, 1)
        
        for i, axis in enumerate(['x', 'y', 'z'][:acc_data.shape[1]]):
            signal = acc_data[:, i]
            features[f'acc_{axis}_mean'] = np.mean(signal)
            features[f'acc_{axis}_std'] = np.std(signal)
        
        if acc_data.shape[1] >= 3:
            magnitude = np.sqrt(np.sum(acc_data[:, :3]**2, axis=1))
            features['acc_magnitude_mean'] = np.mean(magnitude)
            features['movement_activity'] = np.sum(magnitude > np.mean(magnitude) + np.std(magnitude))
        else:
            features['acc_magnitude_mean'] = 0
            features['movement_activity'] = 0
        
        return features
    
    def process_enhanced_dataset(self, participant_limit=20, files_per_participant=20):
        """Process dataset with task performance integration"""
        print("🚀 Processing Enhanced CLAS Dataset with Task Performance...")
        
        participants = list(range(1, min(participant_limit + 1, 63)))
        all_records = []
        successful_participants = 0
        participants_with_task_data = 0
        
        for participant_id in participants:
            try:
                records = self.process_participant_with_answers(participant_id, files_per_participant)
                if len(records) > 0:
                    all_records.extend(records)
                    successful_participants += 1
                    
                    # Check if this participant has task performance data
                    if any(record.get('has_task_data', False) for record in records):
                        participants_with_task_data += 1
                    
                if successful_participants % 5 == 0:
                    print(f"✅ Processed {successful_participants} participants...")
                    
            except Exception as e:
                print(f"❌ Error processing participant {participant_id}: {e}")
                continue
        
        print(f"\n📊 Enhanced Processing Summary:")
        print(f"  Total participants: {successful_participants}")
        print(f"  Participants with task data: {participants_with_task_data}")
        print(f"  Total records: {len(all_records)}")
        
        if len(all_records) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract features and labels
        feature_list = []
        stress_labels = []
        workload_labels = []
        
        for record in all_records:
            feature_list.append(record['features'])
            stress_labels.append(record['stress_level'])
            workload_labels.append(record['workload_level'])
        
        features_df = pd.DataFrame(feature_list).fillna(0)
        self.feature_names = features_df.columns.tolist()
        
        print(f"  Feature count: {len(self.feature_names)}")
        task_features = [f for f in self.feature_names if any(keyword in f for keyword in ['task', 'response', 'accuracy', 'math', 'logic', 'stroop', 'performance'])]
        print(f"  Task performance features: {len(task_features)}")
        if task_features:
            print(f"    Task features: {task_features[:5]}...")  # Show first 5
        
        return features_df.values, np.array(stress_labels), np.array(workload_labels)
    
    def train_enhanced_models(self, X, stress_labels, workload_labels):
        """Train models with enhanced features"""
        print(f"\n🧠 Training Enhanced Models...")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        
        if X.shape[0] < 10:
            print("⚠️  Warning: Very small dataset. Results may not be reliable.")
            print("   Recommend processing more participants for better results.")
        
        if X.shape[0] < 4:
            print("❌ Dataset too small for training. Need at least 4 samples.")
            return {
                'stress_accuracy': 0.0,
                'workload_accuracy': 0.0,
                'stress_model': None,
                'workload_model': None
            }
        
        try:
            # For very small datasets, use simple train-test split or no split
            if X.shape[0] < 20:
                # Simple holdout validation for small datasets
                test_size = max(0.2, 2/X.shape[0])  # At least 2 samples for test
                
                # Stress model
                if len(np.unique(stress_labels)) > 1:
                    X_train, X_test, y_train_stress, y_test_stress = train_test_split(
                        X, stress_labels, test_size=test_size, random_state=42)
                    
                    scaler_stress = StandardScaler()
                    X_train_scaled = scaler_stress.fit_transform(X_train)
                    X_test_scaled = scaler_stress.transform(X_test)
                    
                    # Use simpler model for small datasets
                    rf_stress = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                    rf_stress.fit(X_train_scaled, y_train_stress)
                    
                    y_pred_stress = rf_stress.predict(X_test_scaled)
                    stress_accuracy = accuracy_score(y_test_stress, y_pred_stress)
                else:
                    stress_accuracy = 0.0
                    rf_stress = None
                
                # Workload model
                if len(np.unique(workload_labels)) > 1:
                    X_train, X_test, y_train_workload, y_test_workload = train_test_split(
                        X, workload_labels, test_size=test_size, random_state=42)
                    
                    scaler_workload = StandardScaler()
                    X_train_scaled = scaler_workload.fit_transform(X_train)
                    X_test_scaled = scaler_workload.transform(X_test)
                    
                    rf_workload = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                    rf_workload.fit(X_train_scaled, y_train_workload)
                    
                    y_pred_workload = rf_workload.predict(X_test_scaled)
                    workload_accuracy = accuracy_score(y_test_workload, y_pred_workload)
                else:
                    workload_accuracy = 0.0
                    rf_workload = None
                    
            else:
                # Original approach for larger datasets
                X_train, X_test, y_train_stress, y_test_stress = train_test_split(
                    X, stress_labels, test_size=0.2, random_state=42, stratify=stress_labels)
                
                scaler_stress = StandardScaler()
                X_train_scaled = scaler_stress.fit_transform(X_train)
                X_test_scaled = scaler_stress.transform(X_test)
                
                rf_stress = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
                rf_stress.fit(X_train_scaled, y_train_stress)
                
                y_pred_stress = rf_stress.predict(X_test_scaled)
                stress_accuracy = accuracy_score(y_test_stress, y_pred_stress)
                
                # Workload model
                X_train, X_test, y_train_workload, y_test_workload = train_test_split(
                    X, workload_labels, test_size=0.2, random_state=42, stratify=workload_labels)
                
                scaler_workload = StandardScaler()
                X_train_scaled = scaler_workload.fit_transform(X_train)
                X_test_scaled = scaler_workload.transform(X_test)
                
                rf_workload = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
                rf_workload.fit(X_train_scaled, y_train_workload)
                
                y_pred_workload = rf_workload.predict(X_test_scaled)
                workload_accuracy = accuracy_score(y_test_workload, y_pred_workload)
            
            print(f"\n🎯 Enhanced Stress Recognition:")
            print(f"  Accuracy: {stress_accuracy:.3f}")
            print(f"  Class distribution: {dict(zip(*np.unique(stress_labels, return_counts=True)))}")
            
            print(f"\n🧠 Enhanced Workload Assessment:")
            print(f"  Accuracy: {workload_accuracy:.3f}")
            print(f"  Class distribution: {dict(zip(*np.unique(workload_labels, return_counts=True)))}")
            
            return {
                'stress_accuracy': stress_accuracy,
                'workload_accuracy': workload_accuracy,
                'stress_model': rf_stress,
                'workload_model': rf_workload
            }
            
        except Exception as e:
            print(f"❌ Error in model training: {e}")
            return {
                'stress_accuracy': 0.0,
                'workload_accuracy': 0.0,
                'stress_model': None,
                'workload_model': None
            }

def main_enhanced():
    """Enhanced main function with task performance integration"""
    data_path = '/kaggle/input/clasnit/CLAS_Database/CLAS'
    
    model = EnhancedCLASModelWithAnswers(data_path)
    
    print("🚀 Starting Enhanced CLAS Processing with Task Performance...")
    X, stress_labels, workload_labels = model.process_enhanced_dataset(
        participant_limit=15,
        files_per_participant=15
    )
    
    if X.shape[0] == 0:
        print("❌ No data processed.")
        return None
    
    results = model.train_enhanced_models(X, stress_labels, workload_labels)
    
    print(f"\n🏆 ENHANCED RESULTS:")
    print(f"  Stress Recognition: {results['stress_accuracy']:.1%}")
    print(f"  Workload Assessment: {results['workload_accuracy']:.1%}")
    
    return model, results

if __name__ == "__main__":
    enhanced_model, enhanced_results = main_enhanced()
