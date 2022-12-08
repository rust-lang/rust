#!/bin/bash

set -ex

while ./x.py test --test-args=--force-rerun --stage 2 src/test/run-make/coverage-reports
do
    :
done
