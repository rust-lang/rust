#!/bin/bash

set -ex

for i in $(seq 1 300);
do
    python3 ../x.py t --stage 2 --set build.print-step-timings=false ./src/tools/cargo -- jobserver
done
