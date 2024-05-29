ckpt_dir=/home/nsccgz_qylin_1/checkpoint

raw_data_file="/mnt/ssd/dataset/kaggle/train.txt"
processd_data="/mnt/ssd/dataset/kaggle/kaggleAdDisplayChallenge_processed.npz"

check_freq=10
num_batches=500
perf_out_path=/home/nsccgz_qylin_1/IncrCP_paper/experimental_results/test_concat

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
  --eperc=0 \
  --concat=1 \
  --perf-out-path="$perf_out_path/concat.json"


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
  --eperc=0 \
  --concat=0 \
  --perf-out-path="$perf_out_path/wo_concat.json"