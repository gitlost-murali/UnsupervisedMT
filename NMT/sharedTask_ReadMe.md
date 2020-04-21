`get_data_combined_hsb_de.sh` is for concatenating the both corpora and getting one embedding for Xlingual embeddings, which includes one vocab, one bpe etc.

`get_bpeseparate_hsb_de.sh` creates separate files for each language, i.e seperate bpe codes and vocabs. This doesn't do aligning.

In `line 216` of `MUSE/src/evaluation/evaluator.py`, write the following if you don't have a dictionary:

    if self.params.dico_eval:
       self.word_translation(to_log)

Updated `line 61` of `MUSE/unsupervised.py`
parser.add_argument("--dico_eval", type=str, default="", help="Path to evaluation dictionary. If an empty path is given, then don't use a dictionary to evaluate the aligned word embeddings.")


New code for Unsupervised Bilingual agreement will go in `trainer.py`'s function `otf_bt`. Line `685`