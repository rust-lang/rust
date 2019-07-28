#!/bin/sh

# This file provides the function `commit_toolstate_change` for pushing a change
# to the `rust-toolstate` repository.
#
# The function relies on a GitHub bot user, which should have a Personal access
# token defined in the environment variable $TOOLSTATE_REPO_ACCESS_TOKEN. If for
# some reason you need to change the token, please update `.azure-pipelines/*`.
#
#   1. Generate a new Personal access token:
#
#       * Login to the bot account, and go to Settings -> Developer settings ->
#           Personal access tokens
#       * Click "Generate new token"
#       * Enable the "public_repo" permission, then click "Generate token"
#       * Copy the generated token (should be a 40-digit hexadecimal number).
#           Save it somewhere secure, as the token would be gone once you leave
#           the page.
#
#   2. Update the variable group in Azure Pipelines
#
#       * Ping a member of the infrastructure team to do this.
#
#   4. Replace the email address below if the bot account identity is changed
#
#       * See <https://help.github.com/articles/about-commit-email-addresses/>
#           if a private email by GitHub is wanted.

commit_toolstate_change() {
    OLDFLAGS="$-"
    set -eu

    git config --global user.email '7378925+rust-toolstate-update@users.noreply.github.com'
    git config --global user.name 'Rust Toolstate Update'
    git config --global credential.helper store
    printf 'https://%s:x-oauth-basic@github.com\n' "$TOOLSTATE_REPO_ACCESS_TOKEN" \
        > "$HOME/.git-credentials"
    git clone --depth=1 $TOOLSTATE_REPO

    cd rust-toolstate
    FAILURE=1
    MESSAGE_FILE="$1"
    shift
    for RETRY_COUNT in 1 2 3 4 5; do
        # Call the callback.
        # - If we are in the `auto` branch (pre-landing), this is called from `checktools.sh` and
        #   the callback is `change_toolstate` in that file. The purpose of this is to publish the
        #   test results (the new commit-to-toolstate mapping) in the toolstate repo.
        # - If we are in the `master` branch (post-landing), this is called by the CI pipeline
        #   and the callback is `src/tools/publish_toolstate.py`. The purpose is to publish
        #   the new "current" toolstate in the toolstate repo.
        "$@"
        # `git commit` failing means nothing to commit.
        FAILURE=0
        git commit -a -F "$MESSAGE_FILE" || break
        # On failure randomly sleep for 0 to 3 seconds as a crude way to introduce jittering.
        git push origin master && break || sleep $(LC_ALL=C tr -cd 0-3 < /dev/urandom | head -c 1)
        FAILURE=1
        git fetch origin master
        git reset --hard origin/master
    done
    cd ..

    set +eu
    set "-$OLDFLAGS"
    return $FAILURE
}
