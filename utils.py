import numpy as np

def check_missing(data):
    return data.isnull().sum()

def detect_outliers(data):
    numeric = data.select_dtypes(include=np.number)
    z_scores = np.abs((numeric - numeric.mean()) / numeric.std())
    return (z_scores > 3).sum()

def quality_score(data):
    missing = data.isnull().mean().mean()
    duplicates = data.duplicated().mean()

    score = 100 - (missing * 50 + duplicates * 50)
    return round(score, 2)

# Completeness
def completeness(data):
    return 1 - data.isnull().mean().mean()

# Uniqueness
def uniqueness(data):
    return 1 - data.duplicated().mean()

# Validity
def validity(data):
    valid_age = data["age"].between(0,100).mean() if "age" in data.columns else 1
    return valid_age

# Consistency
def consistency(data):
    if "phone" in data.columns:
        return data["phone"].astype(str).str.len().eq(10).mean()
    return 1

# Accuracy (based on outliers)
def accuracy(data):
    numeric = data.select_dtypes(include=np.number)
    if numeric.empty:
        return 1
    z = np.abs((numeric - numeric.mean()) / numeric.std())
    return 1 - (z > 3).mean().mean()

# Integrity
def integrity(data):
    if "id" in data.columns:
        return data["id"].notnull().mean()
    return 1


def overall_quality(data):
    scores = {
        "Completeness": completeness(data),
        "Uniqueness": uniqueness(data),
        "Validity": validity(data),
        "Consistency": consistency(data),
        "Accuracy": accuracy(data),
        "Integrity": integrity(data)
    }
    return scores




def auto_fix(data):

    fixed_data = data.copy()

    # 1. Fill missing values (numeric → mean)
    for col in fixed_data.select_dtypes(include=np.number).columns:
        fixed_data[col].fillna(fixed_data[col].mean(), inplace=True)

    # 2. Fill missing values (categorical → mode)
    for col in fixed_data.select_dtypes(exclude=np.number).columns:
        fixed_data[col].fillna(fixed_data[col].mode()[0], inplace=True)

    # 3. Remove duplicates
    fixed_data = fixed_data.drop_duplicates()

    # 4. Handle outliers (clip values)
    numeric = fixed_data.select_dtypes(include=np.number)
    for col in numeric.columns:
        mean = numeric[col].mean()
        std = numeric[col].std()
        lower = mean - 3 * std
        upper = mean + 3 * std
        fixed_data[col] = fixed_data[col].clip(lower, upper)

    return fixed_data