import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Housing.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 3. BASIC PREPROCESSING
# ==========================================

# Yes / No columns

yes_no_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for col in yes_no_columns:

    df[col] = df[col].map({
        "yes": 1,
        "no": 0
    })


# ==========================================
# 4. FURNISHING STATUS
# ==========================================

df["furnishingstatus"] = df["furnishingstatus"].map({

    "unfurnished": 0,

    "semi-furnished": 1,

    "furnished": 2

})


# ==========================================
# 5. FEATURES AND TARGET
# ==========================================

X = df.drop("price", axis=1)

y = df["price"]


print("\nFeatures used by model:")

print(X.columns.tolist())


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


print("\nTraining Samples:", len(X_train))

print("Testing Samples:", len(X_test))


# ==========================================
# 7. CREATE DECISION TREE REGRESSOR
# ==========================================

model = DecisionTreeRegressor(

    criterion="squared_error",

    max_depth=5,

    random_state=42

)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


print("\nModel Training Completed!")


# ==========================================
# 9. SAVE HOUSE PRICE MODEL
# ==========================================

joblib.dump(

    model,

    "model.joblib"

)


print("\nHouse Price Model Saved Successfully!")

print("File: model.joblib")


# ==========================================
# 10. VERIFY MODEL FEATURES
# ==========================================

print("\nModel Feature Names:")

print(model.feature_names_in_)