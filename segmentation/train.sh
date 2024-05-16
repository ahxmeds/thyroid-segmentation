torchrun --standalone --nproc_per_node=1 train.py --network-name='unet' --epochs=500 --input-patch-size=128 --train-bs=16 --num_workers=4 --lr=2e-4 --wd=1e-5 --val-interval=2 --sw-bs=2 --cache-rate=1
