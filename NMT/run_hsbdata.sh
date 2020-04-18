#!/bin/bash
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --partition long
#SBATCH --mem-per-cpu=6048
#SBATCH --time=1-10:00:00
##SBATCH --mail-type=END

module add cuda/10.0
module add cudnn/7.3-cuda-10.0



./get_bpeseparate_hsb_de.sh

#python main.py --exp_name test --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'de,hsb' --n_mono -1 --mono_dataset 'de:./data/mono/all.de.tok.60000.pth,,;hsb:./data/mono/all.hsb.tok.60000.pth,,' --para_dataset 'de-hsb:,./data/para/devel.hsb-de.XX.60000.pth,' --mono_directions 'de,hsb' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'hsb-de-hsb,de-hsb-de' --pretrained_emb './data/mono/all.de-hsb.60000.vec' --pretrained_out True --batch_size 16 --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_de_hsb_valid,10

