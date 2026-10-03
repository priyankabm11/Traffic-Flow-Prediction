import pandas as pd

# ============================================
# 1. LOAD PROCESSED DATA
# ============================================

df = pd.read_csv("data/traffic_processed.csv")

print("Dataset shape:", df.shape)

# ============================================
# 2. CALCULATE SPEED STATISTICS
# ============================================

mean_speed = df["speed_kmh"].mean()
std_speed = df["speed_kmh"].std()

print("\nAverage speed:", round(mean_speed, 2), "km/h")
print("Standard deviation:", round(std_speed, 2), "km/h")

# ============================================
# 3. CREATE 5 CONGESTION LEVELS
# ============================================

# Quantile-based classification:
# Lower speed  = higher congestion
# Higher speed = lower congestion

df["speed_rank"] = df["speed_kmh"].rank(method="first")

df["congestion_level"] = pd.qcut(
    df["speed_rank"],
    q=5,
    labels=[4, 3, 2, 1, 0]
).astype(int)

# ============================================
# 4. DISPLAY CONGESTION DISTRIBUTION
# ============================================

print("\nCongestion level distribution:")

print(
    df["congestion_level"]
    .value_counts()
    .sort_index()
)

# ============================================
# 5. DISPLAY SAMPLE DATA
# ============================================

print("\nSample labelled data:")

print(
    df[
        [
            "way",
            "day_of_week",
            "hour",
            "speed_kmh",
            "congestion_level"
        ]
    ].head(10)
)

# ============================================
# 6. REMOVE TEMPORARY COLUMN
# ============================================

df = df.drop(columns=["speed_rank"])

# ============================================
# 7. SAVE LABELLED DATA
# ============================================

df.to_csv(
    "data/traffic_labelled.csv",
    index=False
)

print("\nLabelled dataset saved successfully!")