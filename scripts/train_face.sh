python train.py --name edge2face_512 --instance_feat --netG global_with_features --n_downsample_global 3 --no_instance --label_nc 0 --input_nc 15 --dataroot /vid2vid/dataset --checkpoints_dir /vid2vid/pix2pix_checkpoints --dataset_mode face --loadSize 256 --gpu_ids 0,1 --batchSize 64 --max_dataset_size 6400 --print_freq 10 --display_freq 10 --save_latest_freq 100
