import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path):
    """Memuat data dengan handling khusus untuk NULL values"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di: {path}")
    
    # Membaca 'NULL' teks sebagai NaN
    df = pd.read_csv(path, na_values=['NULL'])
    print(f"[INFO] Data dimuat. Ukuran: {df.shape}")
    return df

def clean_data(df):
    """Membersihkan missing values dan outlier"""
    df = df.copy()
    
    # 1. Hapus Duplikat
    df = df.drop_duplicates()
    
    # 2. Handling Missing Values Spesifik
    if 'company' in df.columns:
        df = df.drop(columns=['company'])
    
    df['agent'] = df['agent'].fillna(0)
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')
    
    # Drop sisa missing values
    df = df.dropna()
    
    # 3. Handling Outlier
    Q1 = df['adr'].quantile(0.25)
    Q3 = df['adr'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df = df[(df['adr'] >= lower) & (df['adr'] <= upper)]
    
    print(f"[INFO] Data setelah cleaning: {df.shape}")
    return df

def feature_engineering(df):
    """Melakukan Binning dan Encoding"""
    df = df.copy()
    
    # 1. Binning 'lead_time'
    bins = [-1, 7, 30, 90, 1000]
    labels = ['Mendadak', 'Pendek', 'Menengah', 'Panjang']
    df['lead_time_group'] = pd.cut(df['lead_time'], bins=bins, labels=labels)
    df = df.drop(columns=['lead_time'])
    
    # 2. Label Encoding
    # Hapus kolom tanggal/ribet, fokus ke fitur utama
    ignore = ['reservation_status_date', 'is_canceled']
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols if c not in ignore]
    
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df

def split_scale_save(df, output_folder="hotel_booking_preprocessing"):
    """Split, Scale, dan Simpan"""
    
    # Fitur & Target
    # Pastikan hanya kolom angka yang masuk (hasil encoding)
    X = df.drop(columns=['is_canceled']).select_dtypes(include=[np.number])
    y = df['is_canceled']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # DataFrame-kan kembali
    X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Simpan
    os.makedirs(output_folder, exist_ok=True)
    
    train_df = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_final, y_test.reset_index(drop=True)], axis=1)
    
    train_df.to_csv(f"{output_folder}/train_processed.csv", index=False)
    test_df.to_csv(f"{output_folder}/test_processed.csv", index=False)
    
    print(f"[SUKSES] Data tersimpan di folder '{output_folder}'")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    DATA_PATH = os.path.join(parent_dir, "hotel_bookings_raw", "hotel_bookings.csv")
    
    print(f"[INFO] Mencari dataset di: {DATA_PATH}")
    
    # Jalankan Pipeline
    df = load_data(DATA_PATH)
    df_clean = clean_data(df)
    df_engineered = feature_engineering(df_clean)
        
    # Simpan output di sebelah script ini
    output_dir = os.path.join(base_dir, "hotel_booking_preprocessing")
    split_scale_save(df_engineered, output_folder=output_dir)
        