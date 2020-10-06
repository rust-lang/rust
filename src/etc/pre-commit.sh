#!/usr/bin/env bash
#
# Call `tidy --bless` before each commit
# Copy this scripts to .git/hooks to activate,
# and remove it from .git/hooks to deactivate.
#

set -Eeuo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)";
COMMAND="$ROOT_DIR/x.py test tidy --bless";

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  COMMAND="python $COMMAND"
fi

echo "Running pre-commit script '$COMMAND'";

cd "$ROOT_DIR"

$COMMAND;
