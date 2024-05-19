#!/usr/bin/env bash
#
# Call `tidy` before git push
# Copy this script to .git/hooks to activate,
# and remove it from .git/hooks to deactivate.
#

set -Euo pipefail

# https://github.com/rust-lang/rust/issues/77620#issuecomment-705144570
unset GIT_DIR
ROOT_DIR="$(git rev-parse --show-toplevel)"

echo "Running pre-push script $ROOT_DIR/x test tidy"

cd "$ROOT_DIR"
./x test tidy --set build.locked-deps=true
if [ $? -ne 0 ]; then
    echo "You may use \`git push --no-verify\` to skip this check."
    exit 1
fi
