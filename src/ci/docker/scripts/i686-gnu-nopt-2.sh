#!/bin/bash

set -ex

python3 ../x.py test --stage 1 --set rust.optimize=false library/std &&
/scripts/stage_2_test_set2.sh
