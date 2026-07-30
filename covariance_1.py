import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 1,000 samples from an exponential distribution (inherently right-skewed)
right_skewed_array = np.random.exponential(scale=2.0, size=1000)

# Log transformation
# When the dataset range is very high
# Square root transformation
# when the datset range is in lower range.

# new = np.log1p(arr)
# print(new)

# Before transormation
import seaborn as sns
sns.histplot(right_skewed_array)

new =np.sqrt(right_skewed_array)

new_log = np.log1p(right_skewed_array)

sns.histplot(new_log)

import pandas as pd

df = pd.DataFrame({
    'Colors':['red', 'green', 'blue', 'green'],
    'Education':['High School', 'Intermediate', 'Bachelors', 'Masters']
})

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[['Colors']])
pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())

# Label Encoder
# Ordinal Encoder
from sklearn.preprocessing import OrdinalEncoder
ord_encoder = OrdinalEncoder(categories=[['High School', 'Intermediate', 'Bachelors', 'Masters']])
encoded_data = ord_encoder.fit_transform(df[['Education']])
df['Encoded_Education'] = encoded_data

df