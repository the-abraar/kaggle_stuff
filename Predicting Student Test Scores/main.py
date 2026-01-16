
import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')


class Config:
    SEED = 42
    N_FOLDS = 10 
    TARGET = 'exam_score'
    ID_COL = 'id'


def load_data():
    train = pd.read_csv('kaggle_stuff/Predicting Student Test Scores/data/train.csv')
    test = pd.read_csv('kaggle_stuff/Predicting Student Test Scores/data/test.csv')
    submission = pd.read_csv('kaggle_stuff/Predicting Student Test Scores/data/sample_submission.csv')
    return train, test, submission


def preprocessing_and_fe(df_train, df_test):
    """
    """
    print("⚡ Starting Feature Engineering...")
    
    # Concatenate for consistent processing
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    # --- 1. Ordinal Encoding (Preserving Order matters!) ---
    # We map these manually because 'low' < 'medium' < 'high' implies mathematical distance.
    
    # Sleep Quality
    sleep_map = {'poor': 0, 'average': 1, 'good': 2, 'excellent': 3} 
    # Handle potential missing/unknown values gracefully
    df['sleep_quality'] = df['sleep_quality'].map(sleep_map).fillna(1) 
    
    # Facility Rating
    facility_map = {'low': 0, 'moderate': 1, 'medium': 1, 'high': 2}
    df['facility_rating'] = df['facility_rating'].map(facility_map).fillna(1)
    
    # Exam Difficulty (Inverted: Harder exam might mean lower score, but it's an ordinal feature)
    diff_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    df['exam_difficulty'] = df['exam_difficulty'].map(diff_map).fillna(1)
    
    # Internet Access
    df['internet_access'] = df['internet_access'].map({'yes': 1, 'no': 0})
    
    # --- 2. Advanced Feature Engineering ---
    
    # Interaction: Effective Study (Hours * Attendance)
    # Rationale: Studying 10 hours but missing all classes is different from 10 hours + 100% attendance.
    df['study_efficiency'] = df['study_hours'] * (df['class_attendance'] / 100)
    
    # Interaction: Sleep Efficiency
    df['rest_factor'] = df['sleep_hours'] * (df['sleep_quality'] + 1)
    
    # Ratio: Study vs Sleep
    df['study_sleep_ratio'] = df['study_hours'] / (df['sleep_hours'] + 0.1)
    
    # --- 3. Categorical Encoding (Nominal) ---
    # For Tree models, Label Encoding is often sufficient and creates fewer sparse columns than OneHot
    cat_cols = ['gender', 'course', 'study_method']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        df[col] = df[col].astype('category')

    # --- 4. Split back ---
    drop_train_cols = ['is_train', Config.ID_COL, Config.TARGET]
    drop_test_cols = ['is_train', Config.ID_COL, Config.TARGET]

    train_processed = df[df['is_train'] == 1].drop(
        columns=[c for c in drop_train_cols if c in df.columns]
    )

    test_processed = df[df['is_train'] == 0].drop(
        columns=[c for c in drop_test_cols if c in df.columns]
    )

    
    return train_processed, test_processed



def train_model(model_cls, train_X, train_y, test_X, params, model_name):
    """
    Generic training loop with K-Fold Cross Validation.
    Returns: OOF Predictions (for Stacking), Test Predictions, and Average RMSE
    """
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    oof_preds = np.zeros(len(train_X))
    test_preds = np.zeros(len(test_X))
    scores = []
    
    print(f"\n Training {model_name}...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_X, train_y)):
        X_train, y_train = train_X.iloc[train_idx], train_y.iloc[train_idx]
        X_val, y_val = train_X.iloc[val_idx], train_y.iloc[val_idx]
        
        if model_name == 'CatBoost':
            model = model_cls(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0, early_stopping_rounds=100)
        elif model_name == 'LGBM':
            model = model_cls(**params)
            # LGBM requires specific callbacks for logging suppression in newer versions
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        elif model_name == 'XGB':
            # XGBoost does NOT support pandas categorical dtype
            X_train_xgb = X_train.copy()
            X_val_xgb = X_val.copy()
            test_X_xgb = test_X.copy()

            for col in X_train_xgb.select_dtypes(include='category').columns:
                X_train_xgb[col] = X_train_xgb[col].cat.codes
                X_val_xgb[col] = X_val_xgb[col].cat.codes
                test_X_xgb[col] = test_X_xgb[col].cat.codes

            model = model_cls(**params)
            model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)

        
        if model_name == 'XGB':
            val_pred = model.predict(X_val_xgb)
            oof_preds[val_idx] = val_pred
            score = np.sqrt(mean_squared_error(y_val, val_pred))
            scores.append(score)
            test_preds += model.predict(test_X_xgb) / Config.N_FOLDS
        else:
            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred
            
            score = np.sqrt(mean_squared_error(y_val, val_pred))
            scores.append(score)
            
            # Add to test prediction (averaging later)
            test_preds += model.predict(test_X) / Config.N_FOLDS

        
    avg_score = np.mean(scores)
    print(f" {model_name} Average RMSE: {avg_score:.5f}")
    return oof_preds, test_preds



def solve():
    # 1. Load
    train_df, test_df, submission = load_data()
    
    # 2. Preprocess
    X, X_test = preprocessing_and_fe(train_df, test_df)
    y = train_df[Config.TARGET]
    
    # 3. Model Hyperparameters (Tuned for general tabular performance)
    
    # XGBoost Params
    xgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'tree_method': 'hist', # Faster
        'random_state': Config.SEED
    }
    
    # LightGBM Params
    lgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': Config.SEED,
        'verbosity': -1
    }
    
    # CatBoost Params
    cat_params = {
        'iterations': 2000,
        'learning_rate': 0.01,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'random_seed': Config.SEED,
        'cat_features': ['gender', 'course', 'study_method'], # CatBoost handles these natively well
        'verbose': 0
    }

    # 4. Train Individual Models (Level 1)
    oof_xgb, preds_xgb = train_model(xgb.XGBRegressor, X, y, X_test, xgb_params, 'XGB')
    oof_lgb, preds_lgb = train_model(lgb.LGBMRegressor, X, y, X_test, lgb_params, 'LGBM')
    
    # Note: For CatBoost, passing indices of cat features is beneficial
    # We rely on the Pool or automatic detection usually, but here we pass params.
    # To be perfectly safe with sklearn API:
    oof_cat, preds_cat = train_model(CatBoostRegressor, X, y, X_test, cat_params, 'CatBoost')
    
    # 5. Ensemble (Level 2 - Weighted Blending)
    # Instead of a complex meta-learner, we use weighted averaging based on inverse error.
    # However, a simple geometric blend often works best for High correlation models.
    
    # Let's use a dynamic weight optimization (Simple Linear Regression on OOF)
    from sklearn.linear_model import Ridge
    
    print("\n🧠 Stacking Models...")
    stack_X_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
    stack_X_test = np.column_stack([preds_xgb, preds_lgb, preds_cat])
    
    # Meta Learner - Ridge is robust against overfitting
    meta_model = Ridge(alpha=10)
    meta_model.fit(stack_X_train, y)
    
    final_predictions = meta_model.predict(stack_X_test)
    
    # Clip predictions to realistic bounds (0-100)
    final_predictions = np.clip(final_predictions, 0, 100)
    
    ensemble_score = np.sqrt(mean_squared_error(y, meta_model.predict(stack_X_train)))
    print(f" Ensemble OOF RMSE: {ensemble_score:.5f}")
    
    # 6. Submission
    submission[Config.TARGET] = final_predictions
    submission.to_csv('ubmission.csv', index=False)
    print("\n💾 Submission saved to 'submission.csv'")




if __name__ == "__main__":
    solve()