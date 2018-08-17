#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/mnist/lenet_solver_finetune.prototxt --weights=examples/mnist/lenet_iter_5000_pruned.caffemodel $@
