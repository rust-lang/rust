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

# Check if the local working dir contains uncommitted changes (including untracked files).
if [ -n "$(git status --porcelain)" ]; then
    echo "Stashing local uncommitted changes before running Tidy."
    # Stash uncommitted changes so that tidy only checks what you are going to push.
    git stash push -u -q
    if [ $? -ne 0 ]; then
        echo "Error: Failed to stash changes."
        echo "You may use \`git push --no-verify\` to skip this pre-push check."
        exit 1
    fi
    STASHED=true
else
    STASHED=false
fi

./x test tidy --set build.locked-deps=true
TIDY_RESULT=$?

# Only pop the stash if something was previously stashed during this check.
if [ "$STASHED" = true ]; then
    echo "Restoring stashed changes."
    # Split the stash pop into `apply` and `drop` so a user can fix things if this fails.
    if ! git stash apply -q; then
        echo "Warning: Failed to apply stashed changes due to conflicts."
        echo "Please resolve the conflicts manually and then run 'git stash drop' when finished."
    else
        git stash drop -q
    fi
fi

if [ $TIDY_RESULT -ne 0 ]; then
    echo "You may use \`git push --no-verify\` to skip this check."
    exit 1
fi
