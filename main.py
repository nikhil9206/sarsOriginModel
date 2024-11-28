from Bio import SeqIO
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn import model_selection, linear_model

# data_path = 'https://drive.google.com/uc?id=1f1CtRwSohB7uaAypn8iA4oqdXlD_xXL1'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20DNA%20Detectives/SARS_CoV_2_sequences_global.fasta'
cov2_sequences = 'SARS_CoV_2_sequences_global.fasta'

n_sequences = len(sequences)
print(f"There are {n_sequences} sequences")

sequence_1 = np.array(sequences[0])
sequence_10 = np.array(sequences[9])
percent_similarity = np.mean(sequence_1 == sequence_10) * 100
print("Sequence 1 and 10 similarity: %", percent_similarity)

reference = np.array(sequences[0])
mutations_per_seq = i for i in sequences np.mean(refrence!=i)

plt.hist(mutations_per_seq)
plt.xlabel('# mutations')
plt.ylabel('# sequences')
plt.show()

n_sequences_with_N = 0

print(f'{n_sequences_with_N} sequences have at least 1 "N"!')

# Note: This can take a couple minutes to run!
# but we can monitor our progress using the tqdm library (which creates a progress bar)
n_bases_in_seq = len(sequences[0])
columns = {}

# Iterate though all positions in this sequence.
for location in tqdm.tqdm(range(n_bases_in_seq)): # tqdm is a nice library that prints our progress.
  bases_at_location = np.array([s[location] for s in sequences])
  # If there are no mutations at this position, move on.
  if len(set(bases_at_location))==1: continue
  for base in ['A', 'T', 'G', 'C', '-']:
    feature_values = (bases_at_location==base)

    # Set the values of any base that equals 'N' to np.nan.
    feature_values[bases_at_location==['N']] = np.nan

    # Convert from T/F to 0/1.
    feature_values  = feature_values*1

    # Make the column name look like <location>_<base> (1_A, 2_G, 3_A, etc.)
    column_name = str(location) + '_' + base

    # Add column to dict
    columns[column_name] = feature_values


mutation_df = pd.DataFrame(columns)

# Print the size of the feature matrix/table.
n_rows = np.shape(mutation_df)[0]
n_columns = np.shape(mutation_df)[1]
print(f"Size of matrix: {n_rows} rows x {n_columns} columns")

# Check what the matrix looks like:
mutation_df.tail()


### : Replace the Nones below!
countries_to_regions_dict = {
         'Australia': 'Oceania',
         'China': 'Asia',
         'Hong Kong': None,
         'India': None,
         'Nepal': None,
         'South Korea': None,
         'Sri Lanka': None,
         'Taiwan': None,
         'Thailand': None,
         'USA': None,
         'Viet Nam': None
}

regions = [countries_to_regions_dict[c] if c in
           countries_to_regions_dict else 'NA' for c in countries]
mutation_df['label'] = regions


balanced_df = mutation_df.copy()
balanced_df['label'] = regions
balanced_df = balanced_df[balanced_df.label!='NA']
balanced_df = balanced_df.drop_duplicates()
samples_north_america = balanced_df[balanced_df.label == None] ### 
samples_oceania = balanced_df[balanced_df.label == None] ### 
samples_asia = balanced_df[balanced_df.label == None] ### 

# Number of samples we will use from each region.
n = min(len(samples_north_america), None, None) ### 

balanced_df = pd.concat([samples_north_america[:n],
                    samples_asia[:n],
                    samples_oceania[:n]])
print("Number of samples in each region: ", Counter(balanced_df['label']))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


lm = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, solver='saga', random_state=42)

# Split into training/testing set. Use a testing size of 0.2
X_train, X_test, y_train, y_test = ### 

# Train/fit model
### 
# Predict on the test set.
y_pred = None ### 

# Compute accuracy.
accuracy = None ### 
print("Accuracy: %", accuracy)

# Compute confusion matrix.
confusion_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))
confusion_mat.columns = [c + ' predicted' for c in lm.classes_]
confusion_mat.index = [c + ' true' for c in lm.classes_]

print(confusion_mat)
