#!/bin/sh
#
# Call `tidy --bless` before each commit
#
# To enable this hook, run `./x.py run install-git-hook`.
# To disable it, run `./x.py run install-git-hook --remove`
set -Eeuo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel);
COMMAND="$ROOT_DIR/x.py test tidy --bless";

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  COMMAND="python $COMMAND"
fi

echo "Running pre-commit script $COMMAND";

$COMMAND;
