#!/bin/bash

set -eux

export RUST_BACKTRACE="${RUST_BACKTRACE:-full}"
export NEXTEST_STATUS_LEVEL=all

target="${1:-}"

if [ -z "$target" ]; then
    host_target=$(rustc -vV | awk '/^host/ { print $2 }')
    echo "Defaulted to host target $host_target"
    target="$host_target"
fi

if [[ "$target" = *"wasm"* ]]; then
    # Enable the random backend
    export RUSTFLAGS="${RUSTFLAGS:-} --cfg getrandom_backend=\"wasm_js\""
fi

if [ "${USING_CONTAINER_RUSTC:-}" = 1 ]; then
    # Install nonstandard components if we have control of the environment
    rustup target list --installed |
        grep -E "^$target\$" ||
        rustup target add "$target"
fi

# Test our implementation
if [ "${BUILD_ONLY:-}" = "1" ]; then
    echo "no tests to run for build-only targets"
else
    test_builtins=(cargo test --package builtins-test --no-fail-fast --target "$target")
    "${test_builtins[@]}"
    "${test_builtins[@]}" --release
    "${test_builtins[@]}" --features c
    "${test_builtins[@]}" --features c --release
    "${test_builtins[@]}" --features no-asm
    "${test_builtins[@]}" --features no-asm --release
    "${test_builtins[@]}" --features no-f16-f128
    "${test_builtins[@]}" --features no-f16-f128 --release
    "${test_builtins[@]}" --benches
    "${test_builtins[@]}" --benches --release

    if [ "${TEST_VERBATIM:-}" = "1" ]; then
        verb_path=$(cmd.exe //C echo \\\\?\\%cd%\\builtins-test\\target2)
        "${test_builtins[@]}" --target-dir "$verb_path" --features c
    fi
fi

# Ensure there are no duplicate symbols or references to `core` when
# `compiler-builtins` is built with various features. Symcheck invokes Cargo to
# build with the arguments we provide it, then validates the built artifacts.
symcheck=(cargo run -p symbol-check --release)
[[ "$target" = "wasm"* ]] && symcheck+=(--features wasm)
symcheck+=(-- build-and-check)

"${symcheck[@]}" -p compiler_builtins --target "$target"
"${symcheck[@]}" -p compiler_builtins --target "$target" --release
"${symcheck[@]}" -p compiler_builtins --target "$target" --features c
"${symcheck[@]}" -p compiler_builtins --target "$target" --features c --release
"${symcheck[@]}" -p compiler_builtins --target "$target" --features no-asm
"${symcheck[@]}" -p compiler_builtins --target "$target" --features no-asm --release
"${symcheck[@]}" -p compiler_builtins --target "$target" --features no-f16-f128
"${symcheck[@]}" -p compiler_builtins --target "$target" --features no-f16-f128 --release

run_intrinsics_test() {
    args=(
        --target "$target" --verbose \
        --manifest-path builtins-test-intrinsics/Cargo.toml
    )
    args+=( "$@" )

    # symcheck also checks the results of builtins-test-intrinsics
    "${symcheck[@]}" "${args[@]}"

    # FIXME: we get access violations on Windows, our entrypoint may need to
    # be tweaked.
    if [ "${BUILD_ONLY:-}" != "1" ] && ! [[ "$target" = *"windows"* ]]; then
        cargo run "${args[@]}"
    fi
}

# Verify that we haven't dropped any intrinsics/symbols
run_intrinsics_test
run_intrinsics_test --release
run_intrinsics_test --features c
run_intrinsics_test --features c --release

# Verify that there are no undefined symbols to `panic` within our
# implementations
CARGO_PROFILE_DEV_LTO=true run_intrinsics_test
CARGO_PROFILE_RELEASE_LTO=true run_intrinsics_test --release

# Test libm

# Make sure a simple build works
cargo check -p libm --no-default-features --target "$target"

if [ "${MAY_SKIP_LIBM_CI:-}" = "true" ]; then
    echo "skipping libm PR CI"
    exit
fi

mflags=()

# We enumerate features manually.
mflags+=(--no-default-features)

# Enable arch-specific routines when available.
mflags+=(--features arch)

# Always enable `unstable-float` since it expands available API but does not
# change any implementations.
mflags+=(--features unstable-float)

# We need to specifically skip tests for musl-math-sys on systems that can't
# build musl since otherwise `--all` will activate it.
case "$target" in
    # Can't build at all on MSVC, WASM, or thumb
    *windows-msvc*) mflags+=(--exclude musl-math-sys) ;;
    *wasm*) mflags+=(--exclude musl-math-sys) ;;
    *thumb*) mflags+=(--exclude musl-math-sys) ;;

    # We can build musl on MinGW but running tests gets a stack overflow
    *windows-gnu*) ;;
    # FIXME(#309): LE PPC crashes calling the musl version of some functions. It
    # seems like a qemu bug but should be investigated further at some point.
    # See <https://github.com/rust-lang/libm/issues/309>.
    *powerpc64le*) ;;

    # Everything else gets musl enabled
    *) mflags+=(--features libm-test/build-musl) ;;
esac


# Configure which targets test against MPFR
case "$target" in
    # MSVC cannot link MPFR
    *windows-msvc*) ;;
    # FIXME: MinGW should be able to build MPFR, but setup in CI is nontrivial.
    *windows-gnu*) ;;
    # Targets that aren't cross compiled in CI work fine
    aarch64*apple*) mflags+=(--features libm-test/build-mpfr) ;;
    aarch64*linux*) mflags+=(--features libm-test/build-mpfr) ;;
    i586*) mflags+=(--features libm-test/build-mpfr --features gmp-mpfr-sys/force-cross) ;;
    i686*) mflags+=(--features libm-test/build-mpfr) ;;
    x86_64*) mflags+=(--features libm-test/build-mpfr) ;;
esac

# FIXME: `STATUS_DLL_NOT_FOUND` testing macros on CI.
# <https://github.com/rust-lang/rust/issues/128944>
case "$target" in
    *windows-gnu) mflags+=(--exclude libm-macros) ;;
esac

if [ "${BUILD_ONLY:-}" = "1" ]; then
    # If we are on targets that can't run tests, verify that we can build.
    cmd=(cargo build --target "$target" --package libm)
    "${cmd[@]}"
    "${cmd[@]}" --features unstable-intrinsics

    echo "can't run tests on $target; skipping"
else
    mflags+=(--workspace --target "$target")
    cmd=(cargo test "${mflags[@]}")
    profile_flag="--profile"
    
    # If nextest is available, use that
    command -v cargo-nextest && nextest=1 || nextest=0
    if [ "$nextest" = "1" ]; then
        cmd=(cargo nextest run --max-fail=10)

        # Workaround for https://github.com/nextest-rs/nextest/issues/2066
        if [ -f /.dockerenv ]; then
            cfg_file="/tmp/nextest-config.toml"
            echo "[store]" >> "$cfg_file"
            echo "dir = \"$CARGO_TARGET_DIR/nextest\"" >> "$cfg_file"
            cmd+=(--config-file "$cfg_file")
        fi

        # Not all configurations have tests to run on wasm
        [[ "$target" = *"wasm"* ]] && cmd+=(--no-tests=warn)

        cmd+=("${mflags[@]}")
        profile_flag="--cargo-profile"
    fi

    # Test once without intrinsics
    "${cmd[@]}"

    # Run doctests if they were excluded by nextest
    [ "$nextest" = "1" ] && cargo test --doc --exclude compiler_builtins "${mflags[@]}"

    # Exclude the macros and utile crates from the rest of the tests to save CI
    # runtime, they shouldn't have anything feature- or opt-level-dependent.
    cmd+=(--exclude util --exclude libm-macros)

    # Test once with intrinsics enabled
    "${cmd[@]}" --features unstable-intrinsics
    "${cmd[@]}" --features unstable-intrinsics --benches

    # Test the same in release mode, which also increases coverage. Also ensure
    # the soft float routines are checked.
    "${cmd[@]}" "$profile_flag" release-checked
    "${cmd[@]}" "$profile_flag" release-checked --features force-soft-floats
    "${cmd[@]}" "$profile_flag" release-checked --features unstable-intrinsics
    "${cmd[@]}" "$profile_flag" release-checked --features unstable-intrinsics --benches

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
fi
