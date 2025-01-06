#!/bin/sh
# ignore-tidy-linelength

set -eu
set -x # so one can see where we are in the script

X_PY="$1"

# XXX(DEBUG): fail quickly!
python3 "$X_PY" test --stage 1 src/tools/rustfmt
