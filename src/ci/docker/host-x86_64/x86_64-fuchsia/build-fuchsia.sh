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

cd fuchsia

# Run the script inside the Fuchsia checkout responsible for building Fuchsia.
# You can change arguments to the build by setting KEEP_CHECKOUT=1 above and
# modifying them in build_fuchsia_from_rust_ci.sh.
bash scripts/rust/build_fuchsia_from_rust_ci.sh
