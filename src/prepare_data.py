import pandas as pd

# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_excel("data/traffic_data.xlsx")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# ============================================
# 2. CHECK MISSING VALUES
# ============================================

print("\nMissing values:")
print(df.isnull().sum())

# ============================================
# 3. CONVERT UTC TIMESTAMP
# ============================================

df["datetime_utc"] = pd.to_datetime(
    df["datetime_utc"],
    utc=True
)

# ============================================
# 4. CONVERT UTC TO NEW YORK LOCAL TIME
# ============================================

df["datetime_ny"] = (
    df["datetime_utc"]
    .dt.tz_convert("America/New_York")
)

# ============================================
# 5. EXTRACT DAY OF WEEK AND HOUR
# ============================================

df["day_of_week"] = df["datetime_ny"].dt.dayofweek

df["hour"] = df["datetime_ny"].dt.hour

# ============================================
# 6. CALCULATE AVERAGE SPEED
# ============================================

df["speed_kmh"] = (
    (df["distance(meters)"] / 1000)
    / (df["duration(minutes)"] / 60)
)

# ============================================
# 7. DISPLAY PROCESSED DATA
# ============================================

print("\nProcessed data:")

print(
    df[
        [
            "way",
            "distance(meters)",
            "duration(minutes)",
            "datetime_ny",
            "day_of_week",
            "hour",
            "speed_kmh"
        ]
    ].head()
)

# ============================================
# 8. SAVE PROCESSED DATA
# ============================================

df.to_csv(
    "data/traffic_processed.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")