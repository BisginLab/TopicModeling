import json
import logging
import os
import sys

# import modules from strutopy
sys.path.append('C:\\Users\\DELL\\Desktop\\CSC501\\strutopy-main\\src')

import numpy as np
import pandas as pd
from gensim import corpora
from joblib import Parallel, delayed

from modules.chunk_it import chunkIt
from modules.stm import STM

# Create log directory if it doesn't exist
log_dir = "logfiles"
os.makedirs(log_dir, exist_ok=True)

# Logging setup
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename=os.path.join(log_dir, "fit_reference_model.log"),
    encoding="utf-8",
    level=logging.INFO,
)

# Specify root directory
ARTIFACTS_ROOT_DIR = r"C:\Users\DELL\Desktop\CSC501\OneDrive-2024-06-06\Data of tweets"

# Load reference corpus & corresponding dictionary
data = pd.read_csv(f"{ARTIFACTS_ROOT_DIR}/corpus_preproc_tweets.csv")
corpus = corpora.MmCorpus(f"{ARTIFACTS_ROOT_DIR}/BoW_corpus_tweets.mm")
dictionary = corpora.Dictionary.load(f"{ARTIFACTS_ROOT_DIR}/dictionary.mm")

# Print debug information about corpus and dictionary
print(f"Corpus loaded with {len(corpus)} documents.")
print(f"Dictionary loaded with {len(dictionary)} tokens.")

# Additional detailed checks
for i, doc in enumerate(corpus):
    if doc is None:
        print(f"Document {i} in corpus is None.")
    if any(word_id is None for word_id, freq in doc):
        print(f"Document {i} contains a None word_id.")
    if any(freq is None for word_id, freq in doc):
        print(f"Document {i} contains a None frequency.")

for token_id, token in dictionary.items():
    if token_id is None:
        print(f"Token id {token_id} in dictionary is None.")
    if token is None:
        print(f"Token {token} in dictionary is None.")

# Specify metadata as topical prevalence covariances
metadata_columns = ['username', 'location', 'year', 'month', 'day', 'retweets', 'favorites', 'replies']

# Check for any None or NaN values in the metadata array
if np.any(pd.isnull(data[metadata_columns])):
    print("Missing values found in metadata columns:")
    print(data[metadata_columns].isnull().sum())
    data[metadata_columns] = data[metadata_columns].fillna(0)  # Fill missing values with 0 or appropriate value
else:
    print("No missing values in metadata columns")

xmat = np.array(data[metadata_columns])
print(f"xmat shape: {xmat.shape}")

SEED = 12345  # random seed
np.random.seed(SEED)

# Function to check if an object is None
def check_none(obj, name):
    if obj is None:
        raise ValueError(f"{name} is None")
    return obj

# Fit the model for K = 10,...,100 for a fixed seed (with spectral initialisation)
# and save it to artifacts/reference_model/K
k_values = range(10, 101, 10)  # From 10 to 100, in steps of 10

def fit_reference_model(K):
    output_dir = f"{ARTIFACTS_ROOT_DIR}/reference_model/{K}"
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Fit STM on the reference corpus assuming {K} topics")
    kappa_interactions = True  # Metadata influences topic prevalence
    lda_beta = True  # no topical content
    beta_index = None  # no topical content
    max_em_iter = 25  # maximum number of iterations for the EM-algorithm
    sigma_prior = 0  # prior on sigma, for update of the global covariance matrix
    convergence_threshold = 1e-5  # convergence threshold, in accordance with Roberts et al.

    stm_config = {
        "init_type": "random",
        "model_type": "STM",
        "K": K,
        "convergence_threshold": convergence_threshold,
        "lda_beta": lda_beta,
        "max_em_iter": max_em_iter,
        "kappa_interactions": kappa_interactions,
        "sigma_prior": sigma_prior,
        "content": True,  # Set true to utilize metadata
    }

    try:
        # Print debug information before fitting the model
        print(f"Fitting model for K={K}")
        print(f"STM configuration: {stm_config}")

        # Check if any input is None
        check_none(corpus, "corpus")
        check_none(dictionary, "dictionary")
        check_none(xmat, "xmat")
        check_none(stm_config, "stm_config")

        # Fit STM on the reference corpora with the settings specified above
        model = STM(documents=corpus, dictionary=dictionary, X=xmat, **stm_config)
        model.expectation_maximization(saving=True, output_dir=output_dir)

        logging.info(f"Save model to {output_dir}/stm_config.json")
        stm_config_path = os.path.join(output_dir, "stm_config.json")

        # Bookkeep corpus settings if input data changes
        stm_config.update({
            "length_dictionary": len(dictionary),
            "number_of_docs": len(corpus),
        })

        with open(stm_config_path, "w") as f:
            json.dump(stm_config, f)
    except Exception as e:
        logging.error(f"Error fitting model for K={K}: {e}")
        print(f"Error fitting model for K={K}: {e}")

# specify number of cores to use
cores_to_use = 8
# split according to maximal cores_to_use
t_split = chunkIt(list(k_values), cores_to_use)

for ll in range(len(t_split)):
    with Parallel(n_jobs=len(t_split[ll]), verbose=51) as parallel:
        parallel(delayed(fit_reference_model)(K=k) for k in t_split[ll])
