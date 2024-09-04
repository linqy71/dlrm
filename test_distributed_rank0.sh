#!/bin/bash

incrcp=$1
diff=$2
naive_incre=$3

ckpt_dir=/GLOBALFS/nsccgz_zgchen_3/lqy/checkpoints/vary_cf/node0  # directory to save checkpoints
raw_data_file="/GLOBALFS/nsccgz_zgchen_3/lqy/datasets/Terabyte/day"   # kaggle dataset path
processd_data="/GLOBALFS/nsccgz_zgchen_3/lqy/datasets/Terabyte/terabyte_processed.npz"  # kaggle dataset path
check_freq=$4   # checkpoint frequency: number of iterations
num_batches=$5  # numebr of total training iterations
perf_out_path=/GLOBALFS/nsccgz_zgchen_3/lqy/IncrCP_paper/experimental_results/distributed/cf_e0003_$check_freq/node0 # output path
batch_size=1024
master_addr="99.72.4.19"

mkdir -p $perf_out_path

if [ $incrcp = 1 ]; then
  rm -rf $ckpt_dir/incrcp
  mkdir -p $ckpt_dir/incrcp

  python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node-rank=0 --master_addr=$master_addr --master_port=1234 \
        dlrm_s_pytorch_ckpt.py --arch-sparse-feature-size=128 \
        --arch-mlp-bot="13-512-256-128" \
        --arch-mlp-top="512-512-256-1" \
        --max-ind-range=200000000 \
        --data-generation=dataset \
        --data-set=terabyte \
        --raw-data-file=$raw_data_file \
        --processed-data-file=$processd_data \
        --loss-function=bce --round-targets=True --learning-rate=1.0 \
        --mini-batch-size=$batch_size \
        --print-freq=2048 \
        --print-time --test-freq=102400 \
        --test-mini-batch-size=16384 --test-num-workers=16 \
        --memory-map \
        --use-gpu \
        --num-batches=$num_batches \
        --ckpt-method="incrcp" \
        --ckpt-freq=$check_freq \
        --ckpt-dir=$ckpt_dir \
        --dist-backend=nccl \
        --perf-out-path="$perf_out_path" \
        --eperc=0.0003 \
        --concat=1 \
        --incrcp-reset-thres=100
fi


if [ $naive_incre = 1 ]; then
  rm -rf $ckpt_dir/naive_incre
  mkdir -p $ckpt_dir/naive_incre

  python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node-rank=0 --master_addr=$master_addr --master_port=1234 \
        dlrm_s_pytorch_ckpt.py --arch-sparse-feature-size=128 \
        --arch-mlp-bot="13-512-256-128" \
        --arch-mlp-top="512-512-256-1" \
        --max-ind-range=200000000 \
        --data-generation=dataset \
        --data-set=terabyte \
        --raw-data-file=$raw_data_file \
        --processed-data-file=$processd_data \
        --loss-function=bce --round-targets=True --learning-rate=1.0 \
        --mini-batch-size=$batch_size \
        --print-freq=2048 \
        --print-time --test-freq=102400 \
        --test-mini-batch-size=16384 --test-num-workers=16 \
        --memory-map \
        --use-gpu \
        --num-batches=$num_batches \
        --ckpt-method="naive_incre" \
        --ckpt-freq=$check_freq \
        --ckpt-dir=$ckpt_dir \
        --dist-backend=nccl \
        --perf-out-path="$perf_out_path" \
        --eperc=0.02 \
        --concat=1 \
        --incrcp-reset-thres=100
fi

if [ $diff = 1 ]; then
  rm -rf $ckpt_dir/diff
  mkdir -p $ckpt_dir/diff

  python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node-rank=0 --master_addr=$master_addr --master_port=1234 \
        dlrm_s_pytorch_ckpt.py --arch-sparse-feature-size=128 \
        --arch-mlp-bot="13-512-256-128" \
        --arch-mlp-top="512-512-256-1" \
        --max-ind-range=200000000 \
        --data-generation=dataset \
        --data-set=terabyte \
        --raw-data-file=$raw_data_file \
        --processed-data-file=$processd_data \
        --loss-function=bce --round-targets=True --learning-rate=1.0 \
        --mini-batch-size=$batch_size \
        --print-freq=2048 \
        --print-time --test-freq=102400 \
        --test-mini-batch-size=16384 --test-num-workers=16 \
        --memory-map \
        --use-gpu \
        --num-batches=$num_batches \
        --ckpt-method="diff" \
        --ckpt-freq=$check_freq \
        --ckpt-dir=$ckpt_dir \
        --dist-backend=nccl \
        --perf-out-path="$perf_out_path" \
        --eperc=0.02 \
        --concat=1 \
        --incrcp-reset-thres=100
fi