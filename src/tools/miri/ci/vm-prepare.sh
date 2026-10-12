#!/bin/sh
# Prepare an environment in vmactions containers, which don't have many tools
# installed by default.
#
# This file is inspired from <https://github.com/rust-lang/libc>.
# It duplicates a lot of the logic in `.github/workflows/setup/action.yml`.

set -eux

uname -a
case $HOST_TARGET in
    x86_64-unknown-freebsd) deps="pkg install -y libnghttp2 curl bash python" ;;
    *)
      echo "FATAL: unknown VM host target: $HOST_TARGET"
      exit 1
      ;;
esac

# Installs have been flaky so give them a retry.
count=0
success=false
while [ $count -lt 3 ]; do
    $deps && success=true || true
    [ $success = true ] && break
    sleep 3s
    count=$(( count + 1))
done

if [ "$success" != true ]; then
    echo "failed to install dependencies"
    exit 1
fi

# Install rustup and latest stable toolchain.
curl --proto '=https' --tlsv1.2 -sSf --retry 5 https://sh.rustup.rs | sh -s -- \
    --profile minimal \
    --default-toolchain stable \
    --target "$HOST_TARGET" \
    -y

# Source the cargo environment.
. "$HOME/.cargo/env"

# Install the tools we need.
cargo install --locked -f rustup-toolchain-install-master hyperfine

# Install "master" toolchain.
./miri toolchain

# Show Rust version (miri toolchain).
rustup show
rustc -Vv
cargo -V
