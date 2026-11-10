#!/bin/bash

set -ex

for i in $(seq 1 1000);
do
    python3 ../x.py t --stage 2 ./src/tools/cargo -- jobserver
done
