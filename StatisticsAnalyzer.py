import numpy as np

# 1. Generate Data
np.random.seed(42)
data = np.random.randint(10, 100, size=(100, 3))

print(data[:5], "\n")

# 2.Calculate
mean = data.mean(axis=0)
std = data.std(axis=0)
var = data.var(axis=0)
median = np.median(data, axis=0)
minimum = data.min(axis=0)
maximum = data.max(axis=0)

print("Mean:      ", mean)
print("Median:    ", median)
print("Std Dev:   ", std)
print("Variance:  ", var)
print("Min:       ", minimum)
print("Max:       ", maximum, "\n")

# 3. Percentiles (25th, 50th, 75th)

p25 = np.percentile(data, 25, axis=0)
p50 = np.percentile(data, 50, axis=0)
p75 = np.percentile(data, 75, axis=0)

print(" PERCENTILES ")
print("25th percentile:", p25)
print("50th percentile:", p50)
print("75th percentile:", p75, "\n")

# 4. Correlation and Covariance Matrices
corr = np.corrcoef(data.T)
cov = np.cov(data.T)

print(" CORRELATION MATRIX ")
print(corr, "\n")

print(" COVARIANCE MATRIX ")
print(cov, "\n")

# 5. Normalization (Min-Max Scaling)
data_min = data.min(axis=0)
data_max = data.max(axis=0)
normalized = (data - data_min) / (data_max - data_min)

print(" NORMALIZED DATA (first 5 rows) ")
print(normalized[:5], "\n")

# 6. Z-score Standardization
standardized = (data - mean) / std

print(" STANDARDIZED DATA (Z-score, first 5 rows) ")
print(standardized[:5], "\n")

# 7. Summary Report
print("SUMMARY REPORT")
for i in range(3):
    print(f"\nFeature {i+1}:")
    print(f"  Mean       : {mean[i]:.2f}")
    print(f"  Median     : {median[i]:.2f}")
    print(f"  Std Dev    : {std[i]:.2f}")
    print(f"  Variance   : {var[i]:.2f}")
    print(f"  Min        : {minimum[i]}")
    print(f"  Max        : {maximum[i]}")
    print(f"  25% / 50% / 75% : {p25[i]} / {p50[i]} / {p75[i]}")
