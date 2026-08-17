#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python_files=(
    "$PROJECT_ROOT/PrepareData.py"
    "$PROJECT_ROOT/TrainModel.py"
    "$PROJECT_ROOT/PrepareSimulationData.py"
    "$PROJECT_ROOT/Model_Prediction.py"
)

for py_file in "${python_files[@]}"; do
    echo
    echo "=================================================="
    echo "Running: $py_file"
    echo "=================================================="

    uv run python "$py_file"
done
