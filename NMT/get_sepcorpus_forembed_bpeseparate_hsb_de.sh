# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
#
# Data preprocessing configuration
#

N_MONO=10000000  # number of monolingual sentences for each language
CODES=60000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs


#
# Initialize tools and data paths
#

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
EMBED_PATH=$PWD/embed # For alignment
EMBED_DATA_PATH=$PWD/embed/data # For alignment

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH
mkdir -p $EMBED_PATH
mkdir -p $EMBED_DATA_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

#Alignment - MUSE
MUSE_DIR=$PWD/embed/MUSE

# files full paths
EMBED_SRC_RAW=$EMBED_DATA_PATH/all.de
EMBED_SRC_TOK=$EMBED_DATA_PATH/all.de.tok
EMBED_SRC_TOK_CODES=$EMBED_DATA_PATH/all.de.tok.bpe_$CODES
EMBED_TGT_TOK_CODES=$EMBED_DATA_PATH/all.hsb.tok.bpe_$CODES

# files full paths
SRC_RAW=$MONO_PATH/all.de
TGT_RAW=$MONO_PATH/all.hsb
SRC_TOK=$MONO_PATH/all.de.tok
TGT_TOK=$MONO_PATH/all.hsb.tok
SRC_TOK_CODES=$MONO_PATH/all.de.tok.bpe_$CODES
TGT_TOK_CODES=$MONO_PATH/all.hsb.tok.bpe_$CODES

SRC_BPE_CODES=$MONO_PATH/de_bpe_codes
TGT_BPE_CODES=$MONO_PATH/hsb_bpe_codes

#CONCAT_BPE=$MONO_PATH/all.de-hsb.$CODES
SRC_VOCAB=$MONO_PATH/bpe_vocab.de
TGT_VOCAB=$MONO_PATH/bpe_vocab.hsb
#FULL_VOCAB=$MONO_PATH/vocab.de-hsb.$CODES
SRC_VALID=$PARA_PATH/devel.hsb-de.de
TGT_VALID=$PARA_PATH/devel.hsb-de.hsb
SRC_TEST=$PARA_PATH/devel_test.hsb-de.de
TGT_TEST=$PARA_PATH/devel_test.hsb-de.hsb   # FOR THE SAKE OF BREVITY, I'M STORING TRAIN FILE IN TEST VARIABLE. BECAUSE WE DON'T TEST RIGHT AWAY, I'M DOING THIS. WHILE TESTING, USE DIFF DATA.
SRC_VALID_CODES=$PARA_PATH/devel.hsb-de.de.bpe_$CODES
TGT_VALID_CODES=$PARA_PATH/devel.hsb-de.hsb.bpe_$CODES
SRC_TEST_CODES=$PARA_PATH/devel_test.hsb-de.de.bpe_$CODES
TGT_TEST_CODES=$PARA_PATH/devel_test.hsb-de.hsb.bpe_$CODES
#
# Download and install tools
#

# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download MUSE
cd $EMBED_PATH
if [ ! -d "$MUSE_DIR" ]; then
  echo "Cloning MUSE from GitHub repository..."
  git clone https://github.com/facebookresearch/MUSE.git
fi
echo "MUSE found in: $MUSE_DIR"


# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"

# Download Embedding data

cd $EMBED_DATA_PATH
echo "Downloading German files..."
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.de.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.de.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.de.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.de.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz
# wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.de.shuffled.v2.gz
# wget -c http://data.statmt.org/wmt16/translation-task/news.2015.de.shuffled.gz
# wget -c http://data.statmt.org/wmt17/translation-task/news.2016.de.shuffled.gz
# wget -c http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz

# decompress monolingual data
for FILENAME in news*gz; do
  OUTPUT="${FILENAME::-3}"
  if [ ! -f "$OUTPUT" ]; then
    echo "Decompressing $FILENAME..."
    gunzip -k $FILENAME
  else
    echo "$OUTPUT already decompressed."
  fi
done

# concatenate monolingual data files  #[[ -f "$EMBED_SRC_RAW"  ||  -f "$EMBED_SRC_TOK_CODES.vec" ]]
if ! [[ -f "$EMBED_SRC_RAW"  ||  -f "$EMBED_SRC_TOK_CODES.vec" ]]; then
  echo "Concatenating monolingual data..."
  cat $(ls news*de* | grep -v gz) > $EMBED_SRC_RAW
fi
echo "DE monolingual data concatenated in: $EMBED_SRC_RAW"

# tokenize data
if ! [[ -f "$EMBED_SRC_TOK" || -f "$EMBED_SRC_TOK_CODES.vec" ]]; then
  echo "Tokenize monolingual data..."
  cat $EMBED_SRC_RAW | $NORM_PUNC -l de | $TOKENIZER -l de -no-escape -threads $N_THREADS > $EMBED_SRC_TOK
fi
echo "DE monolingual data tokenized in: $EMBED_SRC_TOK"

# learn BPE codes
if [ ! -f "$SRC_BPE_CODES" ]; then
   echo "Learning BPE codes..."
   $FASTBPE learnbpe $CODES $EMBED_SRC_TOK  > $SRC_BPE_CODES
 fi
 echo "BPE learned in $SRC_BPE_CODES"

 # apply BPE codes
if ! [[ -f "$EMBED_SRC_TOK_CODES" || -f "$EMBED_SRC_TOK_CODES.vec" ]]; then
   echo "Applying BPE codes..."
   $FASTBPE applybpe $EMBED_SRC_TOK_CODES $EMBED_SRC_TOK $SRC_BPE_CODES
 fi
 echo "BPE codes applied to DE in: $EMBED_SRC_TOK_CODES"

if ! [[ -f "$EMBED_SRC_TOK_CODES.vec" ]]; then
  echo "Training fastText on $EMBED_SRC_TOK_CODES..."
  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 300 -thread $N_THREADS -ws 5 -neg 10 -input $EMBED_SRC_TOK_CODES -output $EMBED_SRC_TOK_CODES
fi
echo "FastText embeddings of De in: $EMBED_SRC_TOK_CODES.vec"

# concatenate monolingual data files
if [[ -f "$EMBED_SRC_RAW" ]]; then
  echo "Deleting monolingual data..."
  rm $EMBED_SRC_RAW
fi
echo "DE monolingual data deleted i.e: $EMBED_SRC_RAW"

# concatenate monolingual data files
if [[ -f "$EMBED_SRC_TOK" ]]; then
  echo "Deleting monolingual data..."
  rm $EMBED_SRC_TOK
fi
echo "DE monolingual token data deleted i.e: $EMBED_SRC_TOK"


#
# Download monolingual data
#

cd $MONO_PATH

echo "Downloading German files..."
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.de.shuffled.gz
# --here
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz
# wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.de.shuffled.v2.gz
# wget -c http://data.statmt.org/wmt16/translation-task/news.2015.de.shuffled.gz
# wget -c http://data.statmt.org/wmt17/translation-task/news.2016.de.shuffled.gz
# wget -c http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz

echo "Downloading Sorbian files..."
wget -c http://statmt.org/wmt20/unsup_and_very_low_res/sorbian_institute_monolingual.hsb.gz 
wget -c http://statmt.org/wmt20/unsup_and_very_low_res/witaj_monolingual.hsb.gz
wget -c http://statmt.org/wmt20/unsup_and_very_low_res/web_monolingual.hsb.gz


# decompress monolingual data
for FILENAME in news*gz; do
  OUTPUT="${FILENAME::-3}"
  if [ ! -f "$OUTPUT" ]; then
    echo "Decompressing $FILENAME..."
    gunzip -k $FILENAME
  else
    echo "$OUTPUT already decompressed."
  fi
done

# decompress monolingual data
for FILENAME in *hsb*gz; do
  OUTPUT="${FILENAME::-3}"
  if [ ! -f "$OUTPUT" ]; then
    echo "Decompressing $FILENAME..."
    gunzip -k $FILENAME
  else
    echo "$OUTPUT already decompressed."
  fi
done

# concatenate monolingual data files
if ! [[ -f "$SRC_RAW" && -f "$TGT_RAW" ]]; then
  echo "Concatenating monolingual data..."
  cat $(ls news*de* | grep -v gz) | head -n $N_MONO > $SRC_RAW
  cat $(ls *hsb* | grep -v gz) | head -n $N_MONO > $TGT_RAW
fi
echo "DE monolingual data concatenated in: $SRC_RAW"
echo "HSB monolingual data concatenated in: $TGT_RAW"


# # check number of lines
# if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your EN monolingual data."; exit; fi
# if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your FR monolingual data."; exit; fi

# tokenize data
if ! [[ -f "$SRC_TOK" && -f "$TGT_TOK" ]]; then
  echo "Tokenize monolingual data..."
  cat $SRC_RAW | $NORM_PUNC -l de | $TOKENIZER -l de -no-escape -threads $N_THREADS > $SRC_TOK
  cat $TGT_RAW > $TGT_TOK
fi
echo "DE monolingual data tokenized in: $SRC_TOK"
echo "HSB monolingual data tokenized in: $TGT_TOK"

# learn BPE codes
if [ ! -f "$TGT_BPE_CODES" ]; then
   echo "Learning BPE codes..."
#    $FASTBPE learnbpe $CODES $SRC_TOK  > $SRC_BPE_CODES
   $FASTBPE learnbpe $CODES $TGT_TOK  > $TGT_BPE_CODES
 fi
 echo "BPE learned in $SRC_BPE_CODES and $TGT_BPE_CODES "

 # apply BPE codes
if ! [[ -f "$SRC_TOK_CODES" && -f "$TGT_TOK_CODES" ]]; then
   echo "Applying BPE codes..."
   $FASTBPE applybpe $SRC_TOK_CODES $SRC_TOK $SRC_BPE_CODES
   $FASTBPE applybpe $TGT_TOK_CODES $TGT_TOK $TGT_BPE_CODES
 fi
 echo "BPE codes applied to DE in: $SRC_TOK_CODES"
 echo "BPE codes applied to HSB in: $TGT_TOK_CODES"



 ### Murali ==> I have $SRC_TOK.$CODES & $TGT_TOK.$CODES as tokenized texts with words in between.


# extract vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TOK_CODES > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TOK_CODES > $TGT_VOCAB
  # $FASTBPE getvocab $SRC_TOK.$CODES $TGT_TOK.$CODES > $FULL_VOCAB
fi
echo "DE vocab in: $SRC_VOCAB"
echo "HSB vocab in: $TGT_VOCAB"
# echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TOK_CODES.pth" && -f "$TGT_TOK_CODES.pth" ]]; then
  echo "Binarizing data..."
  $UMT_PATH/preprocess.py $SRC_VOCAB $SRC_TOK_CODES
  echo "DE binarized data in: $SRC_TOK_CODES.pth"
  $UMT_PATH/preprocess.py $TGT_VOCAB $TGT_TOK_CODES
fi
echo "DE binarized data in: $SRC_TOK_CODES.pth"
echo "HSB binarized data in: $TGT_TOK_CODES.pth"

#
# Download parallel data (for evaluation only)
#

cd $PARA_PATH

echo "Downloading parallel data..."
wget -c http://www.statmt.org/wmt20/unsup_and_very_low_res/devtest.tar.gz

echo "Extracting parallel data..."
tar -xvzf devtest.tar.gz

# check valid and test files are here
if ! [[ -f "$SRC_VALID" ]]; then echo "$SRC_VALID is not found!"; exit; fi
if ! [[ -f "$TGT_VALID" ]]; then echo "$TGT_VALID is not found!"; exit; fi
if ! [[ -f "$SRC_TEST" ]]; then echo "$SRC_TEST is not found!"; exit; fi
if ! [[ -f "$TGT_TEST" ]]; then echo "$TGT_TEST is not found!"; exit; fi

# echo "Tokenizing valid and test data..."
# $INPUT_FROM_SGM < $SRC_VALID.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID
# $INPUT_FROM_SGM < $TGT_VALID.sgm | $NORM_PUNC -l fr | $REM_NON_PRINT_CHAR | $TOKENIZER -l fr -no-escape -threads $N_THREADS > $TGT_VALID
# $INPUT_FROM_SGM < $SRC_TEST.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST
# $INPUT_FROM_SGM < $TGT_TEST.sgm | $NORM_PUNC -l fr | $REM_NON_PRINT_CHAR | $TOKENIZER -l fr -no-escape -threads $N_THREADS > $TGT_TEST
#

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_VALID_CODES $SRC_VALID $SRC_BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_VALID_CODES $TGT_VALID $TGT_BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $SRC_TEST_CODES $SRC_TEST $SRC_BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_TEST_CODES $TGT_TEST $TGT_BPE_CODES $TGT_VOCAB

echo "Binarizing data..."
rm -f $SRC_VALID.$CODES.pth $TGT_VALID.$CODES.pth $SRC_TEST.$CODES.pth $TGT_TEST.$CODES.pth
 
$UMT_PATH/preprocess.py $SRC_VOCAB $SRC_VALID_CODES
$UMT_PATH/preprocess.py $TGT_VOCAB $TGT_VALID_CODES
$UMT_PATH/preprocess.py $SRC_VOCAB $SRC_TEST_CODES
$UMT_PATH/preprocess.py $TGT_VOCAB $TGT_TEST_CODES
#
#
# #
# # Summary
# #
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    DE: $SRC_TOK_CODES.pth"
echo "    HSB: $TGT_TOK_CODES.pth"
echo "Parallel validation data:"
echo "    DE: $SRC_VALID_CODES.pth"
echo "    HSB: $TGT_VALID_CODES.pth"
# echo "Parallel test data:"
echo "    DE: $SRC_TEST_CODES.pth"
echo "    HSB: $TGT_TEST_CODES.pth"
echo ""

#Embedding for TGT language
if ! [[ -f "$EMBED_TGT_TOK_CODES.vec" ]]; then
  echo "Training fastText on $TGT_TOK_CODES..."
  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 300 -thread $N_THREADS -ws 5 -neg 10 -input $TGT_TOK_CODES -output $EMBED_TGT_TOK_CODES
fi
echo "FastText embeddings of De in: $EMBED_TGT_TOK_CODES.vec"

cd $MUSE_DIR
python unsupervised.py --src_lang de --tgt_lang hsb --src_emb "$EMBED_SRC_TOK_CODES.vec" --tgt_emb "$EMBED_TGT_TOK_CODES.vec" --n_refinement 5 --normalize_embeddings center --exp_path $EMBED_PATH --dis_most_frequent 51255