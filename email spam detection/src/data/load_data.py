# Lifecycle stage 3 — Data Understanding
 
import pandas as pd
 
def load_data(path):
    df = pd.read_csv(path, encoding="cp1252")

     # Keep the two required columns
    df = df[['v1', 'v2']]

    # Rename columns for the spam detection project
    df.columns = ['label', 'message']
    print(df.head())
    print(df.columns)
    print(df["label"].value_counts())

    return df