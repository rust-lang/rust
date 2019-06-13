#!/bin/sh

# This file provides the function `commit_toolstate_change` for pushing a change
# to the `rust-toolstate` repository.
#
# The function relies on a GitHub bot user, which should have a Personal access
# token defined in the environment variable $TOOLSTATE_REPO_ACCESS_TOKEN. If for
# some reason you need to change the token, please update `.travis.yml` and
# `appveyor.yml`:
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
#   2. Encrypt the token for Travis CI
#
#       * Install the `travis` tool locally (`gem install travis`).
#       * Encrypt the token:
#           ```
#           travis -r rust-lang/rust encrypt \
#                   TOOLSTATE_REPO_ACCESS_TOKEN=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
#           ```
#       * Copy output to replace the existing one in `.travis.yml`.
#       * Details of this step can be found in
#           <https://docs.travis-ci.com/user/encryption-keys/>
#
#   3. Encrypt the token for AppVeyor
#
#       * Login to AppVeyor using your main account, and login as the rust-lang
#           organization.
#       * Open the ["Encrypt data" tool](https://ci.appveyor.com/tools/encrypt)
#       * Paste the 40-digit token into the "Value to encrypt" box, then click
#           "Encrypt"
#       * Copy the output to replace the existing one in `appveyor.yml`.
#       * Details of this step can be found in
#           <https://www.appveyor.com/docs/how-to/git-push/>
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
