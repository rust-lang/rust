#!/bin/sh

set -eux

export RUST_BACKTRACE="${RUST_BACKTRACE:-full}"
export NEXTEST_STATUS_LEVEL=all

target="${1:-}"
flags=""

if [ -z "$target" ]; then
    host_target=$(rustc -vV | awk '/^host/ { print $2 }')
    echo "Defaulted to host target $host_target"
    target="$host_target"
fi

# We enumerate features manually.
flags="$flags --no-default-features"

# Enable arch-specific routines when available.
flags="$flags --features arch"

# Always enable `unstable-float` since it expands available API but does not
# change any implementations.
flags="$flags --features unstable-float"

# We need to specifically skip tests for musl-math-sys on systems that can't
# build musl since otherwise `--all` will activate it.
case "$target" in
    # Can't build at all on MSVC, WASM, or thumb
    *windows-msvc*) flags="$flags --exclude musl-math-sys" ;;
    *wasm*) flags="$flags --exclude musl-math-sys" ;;
    *thumb*) flags="$flags --exclude musl-math-sys" ;;

    # We can build musl on MinGW but running tests gets a stack overflow
    *windows-gnu*) ;;
    # FIXME(#309): LE PPC crashes calling the musl version of some functions. It
    # seems like a qemu bug but should be investigated further at some point.
    # See <https://github.com/rust-lang/libm/issues/309>.
    *powerpc64le*) ;;

    # Everything else gets musl enabled
    *) flags="$flags --features libm-test/build-musl" ;;
esac

# Configure which targets test against MPFR
case "$target" in
    # MSVC cannot link MPFR
    *windows-msvc*) ;;
    # FIXME: MinGW should be able to build MPFR, but setup in CI is nontrivial.
    *windows-gnu*) ;;
    # Targets that aren't cross compiled in CI work fine
    aarch64*apple*) flags="$flags --features libm-test/build-mpfr" ;;
    aarch64*linux*) flags="$flags --features libm-test/build-mpfr" ;;
    i586*) flags="$flags --features libm-test/build-mpfr --features gmp-mpfr-sys/force-cross" ;;
    i686*) flags="$flags --features libm-test/build-mpfr" ;;
    x86_64*) flags="$flags --features libm-test/build-mpfr" ;;
esac

# FIXME: `STATUS_DLL_NOT_FOUND` testing macros on CI.
# <https://github.com/rust-lang/rust/issues/128944>
case "$target" in
    *windows-gnu) flags="$flags --exclude libm-macros" ;;
esac

# Make sure we can build with overriding features.
cargo check -p libm --no-default-features

if [ "${BUILD_ONLY:-}" = "1" ]; then
    # If we are on targets that can't run tests, verify that we can build.
    cmd="cargo build --target $target --package libm"
    $cmd
    $cmd --features unstable-intrinsics

    echo "can't run tests on $target; skipping"
    exit
fi

flags="$flags --all --target $target"
cmd="cargo test $flags"
profile="--profile"

# If nextest is available, use that
command -v cargo-nextest && nextest=1 || nextest=0
if [ "$nextest" = "1" ]; then
    # Workaround for https://github.com/nextest-rs/nextest/issues/2066
    if [ -f /.dockerenv ]; then
        cfg_file="/tmp/nextest-config.toml"
        echo "[store]" >> "$cfg_file"
        echo "dir = \"$CARGO_TARGET_DIR/nextest\"" >> "$cfg_file"
        cfg_flag="--config-file $cfg_file"
    fi

    cmd="cargo nextest run ${cfg_flag:-} --max-fail=10 $flags"
    profile="--cargo-profile"
fi

# Test once without intrinsics
$cmd

# Run doctests if they were excluded by nextest
[ "$nextest" = "1" ] && cargo test --doc $flags

# Exclude the macros and utile crates from the rest of the tests to save CI
# runtime, they shouldn't have anything feature- or opt-level-dependent.
cmd="$cmd --exclude util --exclude libm-macros"

# Test once with intrinsics enabled
$cmd --features unstable-intrinsics
$cmd --features unstable-intrinsics --benches

# Test the same in release mode, which also increases coverage. Also ensure
# the soft float routines are checked.
$cmd "$profile" release-checked
$cmd "$profile" release-checked --features force-soft-floats
$cmd "$profile" release-checked --features unstable-intrinsics
$cmd "$profile" release-checked --features unstable-intrinsics --benches

# Ensure that the routines do not panic.
# 
# `--tests` must be passed because no-panic is only enabled as a dev
# dependency. The `release-opt` profile must be used to enable LTO and a
# single CGU.
ENSURE_NO_PANIC=1 cargo build \
     -p libm \
    --target "$target" \
    --no-default-features \
    --features unstable-float \
    --tests \
    --profile release-opt
