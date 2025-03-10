#!/bin/bash

set -ex

sysroot="$(rustc --print sysroot)"
case $OS in
    Linux) export LD_LIBRARY_PATH="$sysroot/lib" ;;
    macOS) export DYLD_FALLBACK_LIBRARY_PATH="$sysroot/lib" ;;
    Windows) export PATH="$(cygpath "$sysroot")/bin:$PATH" ;;
    *) exit 1
esac

# Check sysroot handling
test "$(./target/debug/clippy-driver --print sysroot)" = "$sysroot"

desired_sysroot="target/sysroot"
# Set --sysroot in command line
sysroot=$(./target/debug/clippy-driver --sysroot $desired_sysroot --print sysroot)
test "$sysroot" = $desired_sysroot

# Set --sysroot in arg_file.txt and pass @arg_file.txt to command line
echo "--sysroot=$desired_sysroot" > arg_file.txt
sysroot=$(./target/debug/clippy-driver @arg_file.txt --print sysroot)
test "$sysroot" = $desired_sysroot

# Setting SYSROOT in command line
sysroot=$(SYSROOT=$desired_sysroot ./target/debug/clippy-driver --print sysroot)
test "$sysroot" = $desired_sysroot

# Check that the --sysroot argument is only passed once (SYSROOT is ignored)
(
    cd rustc_tools_util
    touch src/lib.rs
    SYSROOT=/tmp RUSTFLAGS="--sysroot=$(rustc --print sysroot)" ../target/debug/cargo-clippy clippy --verbose
)

# Check that the --sysroot argument is only passed once via arg_file.txt (SYSROOT is ignored)
(
    echo "fn main() {}" > target/driver_test.rs
    echo "--sysroot="$(./target/debug/clippy-driver --print sysroot)"" > arg_file.txt
    echo "--verbose" >> arg_file.txt
    SYSROOT=/tmp ./target/debug/clippy-driver @arg_file.txt ./target/driver_test.rs
)

# Make sure this isn't set - clippy-driver should cope without it
unset CARGO_MANIFEST_DIR

# Run a lint and make sure it produces the expected output. It's also expected to exit with code 1
# FIXME: How to match the clippy invocation in compile-test.rs?
./target/debug/clippy-driver -Dwarnings -Aunused -Zui-testing --emit metadata --crate-type bin tests/ui/char_lit_as_u8.rs 2>char_lit_as_u8.stderr && exit 1
sed -e "/= help: for/d" char_lit_as_u8.stderr > normalized.stderr
diff -u normalized.stderr tests/ui/char_lit_as_u8.stderr

# make sure "clippy-driver --rustc --arg" and "rustc --arg" behave the same
SYSROOT=$(rustc --print sysroot)
diff -u <(./target/debug/clippy-driver --rustc --version --verbose) <(rustc --version --verbose)

echo "fn main() {}" >target/driver_test.rs
# we can't run 2 rustcs on the same file at the same time
CLIPPY=$(./target/debug/clippy-driver ./target/driver_test.rs --rustc)
RUSTC=$(rustc ./target/driver_test.rs)
diff -u <($CLIPPY) <($RUSTC)

# TODO: CLIPPY_CONF_DIR / CARGO_MANIFEST_DIR
