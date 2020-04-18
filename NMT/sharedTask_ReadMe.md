`get_data_combined_hsb_de.sh` is for concatenating the both corpora and getting one embedding for Xlingual embeddings, which includes one vocab, one bpe etc.

`get_bpeseparate_hsb_de.sh` creates separate files for each language, i.e seperate bpe codes and vocabs. This doesn't do aligning.

In `line 216` of `MUSE/src/evaluation/evaluator.py`, write the following if you don't have a dictionary:
    
    if self.params.dico_eval:
       self.word_translation(to_log)