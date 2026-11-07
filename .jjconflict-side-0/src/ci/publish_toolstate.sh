#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/shared.sh"

# The following lines are also found in src/bootstrap/toolstate.rs,
# so if updating here, please also update that file.

export MESSAGE_FILE=$(mktemp -t msg.XXXXXX)

git config --global user.email '7378925+rust-toolstate-update@users.noreply.github.com'
git config --global user.name 'Rust Toolstate Update'
git config --global credential.helper store
printf 'https://%s:x-oauth-basic@github.com\n' "$TOOLSTATE_REPO_ACCESS_TOKEN" \
    > "$HOME/.git-credentials"
git clone --depth=1 $TOOLSTATE_REPO

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_COMMIT_MSG="$(git log --format=%s -n1 HEAD)"

cd rust-toolstate
FAILURE=1
for RETRY_COUNT in 1 2 3 4 5; do
    # The purpose of this is to publish the new "current" toolstate in the toolstate repo.
    # This happens at the end of auto builds.
    # (Publishing the per-commit test results happens in src/bootstrap/toolstate.rs).
    "$(ciCheckoutPath)/src/tools/publish_toolstate.py" "$GIT_COMMIT" \
        "$GIT_COMMIT_MSG" \
        "$MESSAGE_FILE" \
        "$TOOLSTATE_REPO_ACCESS_TOKEN"
    # `git commit` failing means nothing to commit.
    FAILURE=0
    git commit -a -F "$MESSAGE_FILE" || break
    # On failure randomly sleep for 0 to 3 seconds as a crude way to introduce jittering.
    git push origin master && break || sleep $(LC_ALL=C tr -cd 0-3 < /dev/urandom | head -c 1)
    FAILURE=1
    git fetch origin master
    git reset --hard origin/master
done
