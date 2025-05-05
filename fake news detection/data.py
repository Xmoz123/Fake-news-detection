import pandas as pd
import os


os.chdir(r"C:\Users\boipe\OneDrive\Desktop\mini project for google\fake news detection")

df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

df_fake["label"] = 0
df_real["label"] = 1


df = pd.concat([df_fake, df_real], ignore_index=True)


print(df.head())
