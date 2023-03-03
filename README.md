# Domain_Adaptation_Thesis

To Run: 

- place data in ./data folder

- Ensure that it is in format ./data/dataset/domain
    - must have accompanying domain_list.txt file with each line containing a (path, class) pair
    
Train source domain by running image_source.py. 
For example, python image_source.py --trte val --output ckps/source_visda_all/ --da uda --gpu_id 0 --worker 4 --dset VISDA-C --lr 1e-3 --max_epoch 10 --s 0

Perform adaptation by running image_target.py
For example, python image_target_pda.py --cls_par 0.3 --da pda --output_src ckps/source_dslr/ --output ckps/target_pda_dslr_10p/ --gpu_id 0 --dset office --s 1 --worker 4 --batch_size 64

Can use visualize_latent.py to see latent space pre and post adaptation

Can use classification_stats.py to return statistics on adapted model over target dataset. 
