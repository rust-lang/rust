#!/bin/env bash
#
# Call `tidy --bless` before each commit
# Copy this scripts to .git/hooks to activate,
# and remove it from .git/hooks to deactivate.
#
# For help running bash scripts on Windows,
# see https://stackoverflow.com/a/6413405/6894799
#

set -Eeuo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)";
COMMAND="$ROOT_DIR/x.py test tidy --bless";

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  COMMAND="python $COMMAND"
fi

echo "Running pre-commit script $COMMAND";

$COMMAND;
