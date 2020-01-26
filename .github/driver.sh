#! /bin/bash

set -ex

# Check sysroot handling
sysroot=$(./target/debug/clippy-driver --print sysroot)
test "$sysroot" = "$(rustc --print sysroot)"

if [[ ${OS} == "Windows" ]]; then
  desired_sysroot=C:/tmp
else
  desired_sysroot=/tmp
fi
sysroot=$(./target/debug/clippy-driver --sysroot $desired_sysroot --print sysroot)
test "$sysroot" = $desired_sysroot

sysroot=$(SYSROOT=$desired_sysroot ./target/debug/clippy-driver --print sysroot)
test "$sysroot" = $desired_sysroot

# Make sure this isn't set - clippy-driver should cope without it
unset CARGO_MANIFEST_DIR

# Run a lint and make sure it produces the expected output. It's also expected to exit with code 1
# FIXME: How to match the clippy invocation in compile-test.rs?
./target/debug/clippy-driver -Dwarnings -Aunused -Zui-testing --emit metadata --crate-type bin tests/ui/cstring.rs 2> cstring.stderr && exit 1
sed -e "s,tests/ui,\$DIR," -e "/= help/d" cstring.stderr > normalized.stderr
diff normalized.stderr tests/ui/cstring.stderr

# TODO: CLIPPY_CONF_DIR / CARGO_MANIFEST_DIR
