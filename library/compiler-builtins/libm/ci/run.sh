#!/bin/sh

set -eux

export RUST_BACKTRACE="${RUST_BACKTRACE:-full}"
# Needed for no-panic to correct detect a lack of panics
export RUSTFLAGS="${RUSTFLAGS:-} -Ccodegen-units=1"

target="${1:-}"

if [ -z "$target" ]; then
    host_target=$(rustc -vV | awk '/^host/ { print $2 }')
    echo "Defaulted to host target $host_target"
    target="$host_target"
fi

extra_flags=""

# We need to specifically skip tests for musl-math-sys on systems that can't
# build musl since otherwise `--all` will activate it.
case "$target" in
    # Can't build at all on MSVC, WASM, or thumb
    *windows-msvc*) extra_flags="$extra_flags --exclude musl-math-sys" ;;
    *wasm*) extra_flags="$extra_flags --exclude musl-math-sys" ;;
    *thumb*) extra_flags="$extra_flags --exclude musl-math-sys" ;;

    # We can build musl on MinGW but running tests gets a stack overflow
    *windows-gnu*) ;;
    # FIXME(#309): LE PPC crashes calling the musl version of some functions. It
    # seems like a qemu bug but should be investigated further at some point.
    # See <https://github.com/rust-lang/libm/issues/309>.
    *powerpc64le*) ;;

    # Everything else gets musl enabled
    *) extra_flags="$extra_flags --features libm-test/build-musl" ;;
esac

# FIXME: `STATUS_DLL_NOT_FOUND` testing macros on CI.
# <https://github.com/rust-lang/rust/issues/128944>
case "$target" in
    *windows-gnu) extra_flags="$extra_flags --exclude libm-macros" ;;
esac

if [ "$(uname -a)" = "Linux" ]; then
    # also run the reference tests when we can. requires a Linux host.
    extra_flags="$extra_flags --features libm-test/test-musl-serialized"
fi

if [ "${BUILD_ONLY:-}" = "1" ]; then
    cmd="cargo build --target $target --package libm"
    $cmd
    $cmd --features "unstable-intrinsics"

    echo "can't run tests on $target"
else
    cmd="cargo test --all --target $target $extra_flags"

    # stable by default
    $cmd
    $cmd --release

    # unstable with a feature
    $cmd --features "unstable-intrinsics"
    $cmd --release --features "unstable-intrinsics"
fi
