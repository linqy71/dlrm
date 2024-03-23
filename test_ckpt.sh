lsecp=1
diff=1
incre=1
rocksdb=1

ckpt_dir="/mnt/ssd/checkpoint"

raw_data_file="/mnt/ssd/dataset/kaggle/train.txt"
processd_data="/mnt/ssd/dataset/kaggle/kaggleAdDisplayChallenge_processed.npz"

check_freq=10
num_batches=1000

if [ $lsecp=1 ]; then

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
      --ckpt-method="lsecp" \
      --ckpt-freq=$check_freq \
      --ckpt-dir=$ckpt_dir \
      --lsecp-eperc=0.01 \
      --lsecp-clen=10 \
      --perf-out-path="./lsecp.json"

fi


if [ $diff=1 ]; then

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
      --lsecp-eperc=0.01 \
      --lsecp-clen=10 \
      --perf-out-path="./diff.json"

fi

if [ $incre=1 ]; then

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
      --ckpt-method="incre" \
      --ckpt-freq=$check_freq \
      --ckpt-dir=$ckpt_dir \
      --lsecp-eperc=0.01 \
      --lsecp-clen=10 \
      --perf-out-path="./incre.json"

fi

if [ $rocksdb=1 ]; then

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
      --ckpt-method="rocksdb" \
      --ckpt-freq=$check_freq \
      --ckpt-dir=$ckpt_dir \
      --lsecp-eperc=0.01 \
      --lsecp-clen=10 \
      --perf-out-path="./rocksdb.json"

fi