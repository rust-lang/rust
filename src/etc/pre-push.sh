#!/usr/bin/env bash
#
# Call `tidy` before git push
# Copy this script to .git/hooks to activate,
# and remove it from .git/hooks to deactivate.
#

set -Euo pipefail

# Check if the push is doing anything other than deleting remote branches
SKIP=true
while read LOCAL_REF LOCAL_SHA REMOTE_REF REMOTE_SHA; do
    if [[ "$LOCAL_REF" != "(delete)" || \
          "$LOCAL_SHA" != "0000000000000000000000000000000000000000" ]]; then
        SKIP=false
    fi
done

if $SKIP; then
    echo "Skipping tidy check for branch deletion"
    exit 0
fi

ROOT_DIR="$(git rev-parse --show-toplevel)"

echo "Running pre-push script $ROOT_DIR/x test tidy"

cd "$ROOT_DIR"
./x test tidy --set build.locked-deps=true
if [ $? -ne 0 ]; then
    echo "You may use \`git push --no-verify\` to skip this check."
    exit 1
fi
