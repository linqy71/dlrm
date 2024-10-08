incrcp=1
diff=0
naive_incre=1

ckpt_dir=/mnt/ssd/checkpoints  # directory to save checkpoints
raw_data_file="/mnt/ssd/dataset/kaggle/train.txt"   # kaggle dataset path
processd_data="/mnt/ssd/dataset/kaggle/kaggleAdDisplayChallenge_processed.npz"  # kaggle dataset path
check_freq=10   # checkpoint frequency: number of iterations
num_batches=20  # numebr of total training iterations
perf_out_path=/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/test # output path

mkdir -p $perf_out_path

if [ $incrcp = 1 ]; then
  rm -rf $ckpt_dir/incrcp
  mkdir -p $ckpt_dir/incrcp

  python dlrm_s_pytorch_ckpt.py --arch-sparse-feature-size=512 \
      --arch-mlp-bot="13-512-256-64-512" \
      --arch-mlp-top="512-256-1" \
      --max-ind-range=2500000 \
      --data-generation=dataset \
      --data-set=kaggle \
      --raw-data-file=$raw_data_file \
      --processed-data-file=$processd_data \
      --loss-function=bce --round-targets=True --learning-rate=1.0 \
      --mini-batch-size=256 \
      --print-freq=2048 \
      --print-time --test-freq=102400 \
      --test-mini-batch-size=16384 --test-num-workers=16 \
      --mlperf-bin-loader \
      --mlperf-bin-shuffle --use-gpu \
      --num-batches=$num_batches \
      --ckpt-method="incrcp" \
      --ckpt-freq=$check_freq \
      --ckpt-dir=$ckpt_dir \
      --eperc=0.02 \
      --concat=1 \
      --perf-out-path=$perf_out_path \
      --incrcp-reset-thres=100
fi

if [ $diff = 1 ]; then
  rm -rf $ckpt_dir/diff
  mkdir -p $ckpt_dir/diff

  python dlrm_s_pytorch_ckpt.py --arch-sparse-feature-size=512 \
      --arch-mlp-bot="13-512-256-64-512" \
      --arch-mlp-top="512-256-1" \
      --max-ind-range=2500000 \
      --data-generation=dataset \
      --data-set=kaggle \
      --raw-data-file=$raw_data_file \
      --processed-data-file=$processd_data \
      --loss-function=bce --round-targets=True --learning-rate=1.0 \
      --mini-batch-size=256 \
      --print-freq=2048 \
      --print-time --test-freq=102400 \
      --test-mini-batch-size=16384 --test-num-workers=16 \
      --mlperf-bin-loader \
      --mlperf-bin-shuffle --use-gpu \
      --num-batches=$num_batches \
      --ckpt-method="diff" \
      --ckpt-freq=$check_freq \
      --ckpt-dir=$ckpt_dir \
      --perf-out-path=$perf_out_path

fi

if [ $naive_incre = 1 ]; then
  rm -rf $ckpt_dir/naive_incre
  mkdir -p $ckpt_dir/naive_incre

  python dlrm_s_pytorch_ckpt.py --arch-sparse-feature-size=512 \
      --arch-mlp-bot="13-512-256-64-512" \
      --arch-mlp-top="512-256-1" \
      --max-ind-range=2500000 \
      --data-generation=dataset \
      --data-set=kaggle \
      --raw-data-file=$raw_data_file \
      --processed-data-file=$processd_data \
      --loss-function=bce --round-targets=True --learning-rate=1.0 \
      --mini-batch-size=256 \
      --print-freq=2048 \
      --print-time --test-freq=102400 \
      --test-mini-batch-size=16384 --test-num-workers=16 \
      --mlperf-bin-loader \
      --mlperf-bin-shuffle --use-gpu \
      --num-batches=$num_batches \
      --ckpt-method="naive_incre" \
      --ckpt-freq=$check_freq \
      --ckpt-dir=$ckpt_dir \
      --perf-out-path=$perf_out_path 
fi
