#!/usr/bin/env bash

# Downloads and builds the Fuchsia operating system using a toolchain installed
# in $RUST_INSTALL_DIR.
#
# You may run this script locally using Docker with the following command:
#
# $ src/ci/docker/run.sh x86_64-fuchsia
#
# Alternatively, from within the container with --dev, assuming you have made it
# as far as building the toolchain with the above command:
#
# $ src/ci/docker/run.sh --dev x86_64-fuchsia
# docker# git config --global --add safe.directory /checkout/obj/fuchsia
# docker# ../src/ci/docker/host-x86_64/x86_64-fuchsia/build-fuchsia.sh
#
# Also see the docs in the rustc-dev-guide for more info:
# https://github.com/rust-lang/rustc-dev-guide/pull/1989

set -euf -o pipefail

# Set this variable to 1 to disable updating the Fuchsia checkout. This is
# useful for making local changes. You can find the Fuchsia checkout in
# `obj/x86_64-fuchsia/fuchsia` in your local checkout after running this
# job for the first time.
KEEP_CHECKOUT=

# Any upstream refs that should be cherry-picked. This can be used to include
# Gerrit changes from https://fxrev.dev during development (click the "Download"
# button on a changelist to see the cherry pick ref). Example:
# PICK_REFS=(refs/changes/71/1054071/2 refs/changes/74/1054574/2)
PICK_REFS=()

# The commit hash of Fuchsia's integration.git to check out. This controls the
# commit hash of fuchsia.git and some other repos in the "monorepo" checkout, in
# addition to versions of prebuilts. It should be bumped regularly by the
# Fuchsia team – we aim for every 1-2 months.
INTEGRATION_SHA=1c5b42266fbfefb2337c6b2f0030a91bde15f9e9

checkout=fuchsia
jiri=.jiri_root/bin/jiri

set -x

if [ -z "$KEEP_CHECKOUT" ]; then
    # This script will:
    # - create a directory named "fuchsia" if it does not exist
    # - download "jiri" to "fuchsia/.jiri_root/bin"
    curl -s "https://fuchsia.googlesource.com/jiri/+/HEAD/scripts/bootstrap_jiri?format=TEXT" \
        | base64 --decode \
        | bash -s $checkout

    cd $checkout

    $jiri init \
        -partial=true \
        -analytics-opt=false \
        .

    $jiri import \
        -name=integration \
        -revision=$INTEGRATION_SHA \
        -overwrite=true \
        flower \
        "https://fuchsia.googlesource.com/integration"

    if [ -d ".git" ]; then
        # Wipe out any local changes if we're reusing a checkout.
        git checkout --force JIRI_HEAD
    fi

    $jiri update -autoupdate=false

    echo integration commit = $(git -C integration rev-parse HEAD)

    for git_ref in "${PICK_REFS[@]}"; do
        git fetch https://fuchsia.googlesource.com/fuchsia $git_ref
        git cherry-pick --no-commit FETCH_HEAD
    done
else
    echo Reusing existing Fuchsia checkout
    cd $checkout
fi

# Run the script inside the Fuchsia checkout responsible for building Fuchsia.
# You can change arguments to the build by setting KEEP_CHECKOUT=1 above and
# modifying them in build_fuchsia_from_rust_ci.sh.
bash scripts/rust/build_fuchsia_from_rust_ci.sh
