import pandas as pd
import numpy as np
import time
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import numpy as np

# ✅ Load dataset
df = pd.read_csv("smiles_odors.csv")

# ✅ Fix column names
df.columns = df.columns.str.strip()  # Remove extra spaces if any

# ✅ Check required columns
if 'nonStereoSMILES' not in df.columns or 'descriptors' not in df.columns:
    raise KeyError(f"Dataset missing required columns! Found: {df.columns}")

# ✅ Convert SMILES to numerical features
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 10  # Return zeros if invalid molecule
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.RingCount(mol),
        Descriptors.TPSA(mol)
    ]

# ✅ Apply feature extraction
df['features'] = df['nonStereoSMILES'].apply(smiles_to_features)

# ✅ Convert features into NumPy array
X = np.vstack(df['features'].values)

# ✅ Process Odor Labels (MultiLabelBinarizer)
mlb = MultiLabelBinarizer()
df['descriptors'] = df['descriptors'].fillna("")  # Handle missing values
y = mlb.fit_transform(df['descriptors'].apply(lambda x: x.split(';')))  # Split by ';'

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save the model & scaler
joblib.dump(model, "smell_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(mlb, "mlb.pkl")  # Save MultiLabelBinarizer

# ✅ Function for real-time prediction
def real_time_prediction():
    print("\n🔍 Real-time Smell Classification (Press CTRL+C to stop)\n")
    while True:
        smiles_input = input("Enter SMILES string (or 'exit' to stop): ").strip()
        if smiles_input.lower() == 'exit':
            break

        # Convert SMILES to numerical features
        features = np.array(smiles_to_features(smiles_input)).reshape(1, -1)
        features = scaler.transform(features)  # Scale features

        # Predict odors
        predicted_labels = model.predict(features)[0]
        
        detected_odors = mlb.inverse_transform(np.array([predicted_labels]))[0]  # Convert back to names

        # Display results
        print(f"🔬 Predicted Odors: {', '.join(detected_odors) if detected_odors else 'No strong odor detected'}\n")
        time.sleep(2)  # Refresh every 2 seconds

# ✅ Run real-time prediction
if __name__ == "__main__":
    real_time_prediction()
