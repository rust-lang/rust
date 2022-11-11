#!/usr/bin/env bash
#
# Call `tidy --bless` before git push
# Copy this script to .git/hooks to activate,
# and remove it from .git/hooks to deactivate.
#

set -Eeuo pipefail

# https://github.com/rust-lang/rust/issues/77620#issuecomment-705144570
unset GIT_DIR
ROOT_DIR="$(git rev-parse --show-toplevel)"
COMMAND="$ROOT_DIR/x.py test tidy"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  COMMAND="python $COMMAND"
elif ! command -v python &> /dev/null; then
  COMMAND="python3 $COMMAND"
fi

echo "Running pre-push script '$COMMAND'"

cd "$ROOT_DIR"

$COMMAND
