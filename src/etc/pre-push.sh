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


COMMAND="python3.10 $COMMAND"


echo "Running pre-push script '$COMMAND'"

cd "$ROOT_DIR"

$COMMAND
