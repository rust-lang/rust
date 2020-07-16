#!/bin/sh

set -eu

X_PY="$1"

# Try to test all the tools and store the build/test success in the TOOLSTATE_FILE
set +e
python3 "$X_PY" test --no-fail-fast record-toolstate
set -e
# debugging: print out the saved toolstates
cat /tmp/toolstate/toolstates.json
# Check the JSON to ensure that there are no unexpected tool failures.
python3 "$X_PY" test check-tools

# Clippy is not part of the toolstate system;
# just test it the regular way.
python3 "$X_PY" test src/tools/clippy
