# Get directory containing this script
HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR=$HEAD_DIR/code
DATA_DIR=$HEAD_DIR/data
EXP_DIR=$HEAD_DIR/experiments

mkdir -p $EXP_DIR

# Creates the environment
conda create -n qanet python=3.6.6

# Activates the environment
source activate qanet

# Install tensorflow
conda install tensorflow-gpu=1.8.0

# pip install into environment
pip install -r requirements.txt

# download punkt and perluniprops
python -m nltk.downloader punkt
python -m nltk.downloader perluniprops

# Download and preprocess SQuAD data and save in data/
mkdir -p "$DATA_DIR"
rm -rf "$DATA_DIR"
python "$CODE_DIR/preprocessing/squad_preprocess.py" --data_dir "$DATA_DIR"

# Download GloVe vectors to data/
python "$CODE_DIR/preprocessing/download_wordvecs.py" --download_dir "$DATA_DIR"

# Prepare character vocabulary from training context
python "$CODE_DIR/preprocessing/get_char_vocab.py" --data_dir "$DATA_DIR" --data_name "train.context"