### README.md

# SARS-CoV-2 Mutation Analysis and Region Prediction

This project involves analyzing SARS-CoV-2 genomic sequences, extracting mutation features, and training a machine learning model to classify sequences by their region of origin. The pipeline includes sequence alignment, feature extraction, and logistic regression for classification.

---

## Features

1. **Sequence Similarity Analysis**:
   - Compares genomic sequences to calculate mutation rates and similarities.
   - Visualizes the number of mutations across sequences.

2. **Feature Extraction**:
   - Generates a mutation feature matrix based on nucleotide variations at each position in the genome.
   - Handles missing values (e.g., 'N') to ensure robust feature creation.

3. **Region Labeling**:
   - Maps countries to their respective regions (e.g., Asia, Oceania).
   - Balances the dataset by sampling equal numbers of sequences from each region.

4. **Machine Learning Classification**:
   - Uses logistic regression to classify sequences by region of origin.
   - Evaluates model performance using accuracy and confusion matrices.

---

## File Structure

- **`main.py`**: The primary script containing the sequence processing, feature extraction, and machine learning pipeline.
- **`SARS_CoV_2_sequences_global.fasta`**: Input file containing SARS-CoV-2 genomic sequences.

---

## Prerequisites

1. **Python 3.7+**:
   - Ensure Python is installed. Download from [Python Official Website](https://www.python.org/downloads/).

2. **Required Libraries**:
   Install the required Python libraries using pip:
   ```bash
   pip install biopython numpy pandas matplotlib seaborn scikit-learn tqdm
   ```

3. **Input Data**:
   - Download the `SARS_CoV_2_sequences_global.fasta` file.

---

## How to Use

1. **Run the Script**:
   - Execute the script:
     ```bash
     python main.py
     ```

2. **Key Outputs**:
   - Mutation histogram showing the number of mutations per sequence.
   - Feature matrix with mutation data for machine learning.
   - Logistic regression model accuracy and confusion matrix.

---

## Pipeline Description

1. **Data Loading**:
   - Reads SARS-CoV-2 genomic sequences from the provided FASTA file.

2. **Sequence Analysis**:
   - Computes pairwise sequence similarity.
   - Calculates mutation rates per sequence.

3. **Feature Engineering**:
   - Extracts mutations at each genomic position as binary features.
   - Handles missing data and assigns sequence labels based on regions.

4. **Machine Learning**:
   - Balances the dataset by region for fair classification.
   - Splits the data into training and testing sets.
   - Trains a logistic regression model and evaluates its performance.

5. **Evaluation**:
   - Computes model accuracy on the test set.
   - Generates a confusion matrix for detailed performance analysis.

---

## Example Outputs

- **Mutation Histogram**:
  A plot showing the number of mutations across sequences.

- **Confusion Matrix**:
  A detailed table comparing true and predicted labels for sequences.

---

## Future Enhancements

1. Expand the dataset to include more regions and genomic sequences.
2. Explore more advanced models (e.g., Random Forest, Neural Networks).
3. Incorporate additional genomic features for improved accuracy.
4. Optimize feature extraction for faster processing on larger datasets.

---

## Contact

For questions or suggestions, please contact:  
**Email**: [adapalanikhil@gmail.com]  
**Phone**: +1 512-884-7100
```
