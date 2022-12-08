#!/bin/bash

set -ex

make --version

while ./x.py test --test-args=--force-rerun --stage 2 src/test/run-make/coverage-reports
do
    :
done
