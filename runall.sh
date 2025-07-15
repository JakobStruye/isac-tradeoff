#!/bin/bash

# ---------------------
# Default Parameters
# ---------------------
num_epochs_values=(100)
network_values=("CNN")
subsample_time_values=(2 3 4 5 6 7 8 9)
subsample_tx_values=(2 3 4 5 6 7 8 9)
subsample_rx_values=(2 3 4 5 6 7 8 9)
subsample_allbeams_values=(2 3)
subsample_approaches=("repeated")
seeds=$(seq 1 25)

# Run range control
start_count=0
stop_count=-1  # -1 means "no limit"
run_count=0

# ---------------------
# Argument Parsing
# ---------------------
for arg in "$@"; do
  case $arg in
    --start_count=*)
      start_count="${arg#*=}"
      ;;
    --stop_count=*)
      stop_count="${arg#*=}"
      ;;
    --num_epochs=*)
      IFS=',' read -r -a num_epochs_values <<< "${arg#*=}"
      ;;
    --subsample_time=*)
      IFS=',' read -r -a subsample_time_values <<< "${arg#*=}"
      ;;
    --subsample_tx=*)
      IFS=',' read -r -a subsample_tx_values <<< "${arg#*=}"
      ;;
    --subsample_rx=*)
      IFS=',' read -r -a subsample_rx_values <<< "${arg#*=}"
      ;;
    --subsample_allbeams=*)
      IFS=',' read -r -a subsample_allbeams_values <<< "${arg#*=}"
      ;;
    --seeds=*)
      IFS=',' read -r -a seed_list <<< "${arg#*=}"
      seeds="${seed_list[*]}"
      ;;
    --help|-h)
      echo "Usage: $0 [--start_count=N] [--stop_count=M] [--num_epochs=...] [--subsample_time=...] [--subsample_tx=...] [--subsample_rx=...] [--subsample_allbeams=...] [--seeds=...]"
      echo "Values can be comma-separated lists."
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Use --help to see available options."
      exit 1
      ;;
  esac
done

# ---------------------
# Function to run experiments
# ---------------------
run_experiment() {
  local num_epochs=$1
  local network=$2
  local subtime=$3
  local subtx=$4
  local subrx=$5
  local approach=$6
  local approach_flag=""

  if [[ "$approach" != "" ]]; then
    approach_flag="--subsample_approach=$approach"
  fi

  for seed in $seeds; do
    if (( run_count < start_count )); then
      echo "Skipping run #$run_count"
      ((run_count++))
      continue
    fi

    if (( stop_count != -1 && run_count >= stop_count )); then
      echo "Reached stop_count at run #$run_count, exiting."
      exit 0
    fi

    cmd="python traintest.py \
      --num_epochs=$num_epochs \
      --network=$network \
      --seed=$seed \
      --subsample_time=$subtime \
      --subsample_tx=$subtx \
      --subsample_rx=$subrx \
      $approach_flag \
      --log_to_file"

    echo "Running (#$run_count): $cmd"
    eval $cmd
    ((run_count++))
  done
}

# ---------------------
# Experiment Loops
# ---------------------

# Baseline (no subsampling)
for num_epochs in "${num_epochs_values[@]}"; do
  for network in "${network_values[@]}"; do
    run_experiment $num_epochs $network 1 1 1 ""
  done
done

# Time subsampling
for num_epochs in "${num_epochs_values[@]}"; do
  for network in "${network_values[@]}"; do
    for subtime in "${subsample_time_values[@]}"; do
      run_experiment $num_epochs $network $subtime 1 1 ""
    done
  done
done

# TX-only subsampling
for num_epochs in "${num_epochs_values[@]}"; do
  for network in "${network_values[@]}"; do
    for subtx in "${subsample_tx_values[@]}"; do
      for approach in "${subsample_approaches[@]}"; do
        run_experiment $num_epochs $network 1 $subtx 1 $approach
      done
    done
  done
done

# RX-only subsampling
for num_epochs in "${num_epochs_values[@]}"; do
  for network in "${network_values[@]}"; do
    for subrx in "${subsample_rx_values[@]}"; do
      for approach in "${subsample_approaches[@]}"; do
        run_experiment $num_epochs $network 1 1 $subrx $approach
      done
    done
  done
done

# All-beams subsampling
for num_epochs in "${num_epochs_values[@]}"; do
  for network in "${network_values[@]}"; do
    for value in "${subsample_allbeams_values[@]}"; do
      for approach in "${subsample_approaches[@]}"; do
        run_experiment $num_epochs $network 1 $value $value $approach
      done
    done
  done
done
