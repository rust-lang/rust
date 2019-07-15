set -ex

echo "Running clippy base tests"

PATH=$PATH:./node_modules/.bin
if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  remark -f *.md -f doc/*.md > /dev/null
fi
# build clippy in debug mode and run tests
cargo build --features debugging
cargo test --features debugging
# for faster build, share target dir between subcrates
export CARGO_TARGET_DIR=`pwd`/target/
(cd clippy_lints && cargo test)
(cd rustc_tools_util && cargo test)
(cd clippy_dev && cargo test)

# make sure clippy can be called via ./path/to/cargo-clippy
(
  cd clippy_workspace_tests
  ../target/debug/cargo-clippy
)

# Perform various checks for lint registration
./util/dev update_lints --check
./util/dev --limit-stderr-length

# Check running clippy-driver without cargo
(
  # Check sysroot handling
  sysroot=$(./target/debug/clippy-driver --print sysroot)
  test $sysroot = $(rustc --print sysroot)

  if [ -z $OS_WINDOWS ]; then
    desired_sysroot=/tmp
  else
    desired_sysroot=C:/tmp
  fi
  sysroot=$(./target/debug/clippy-driver --sysroot $desired_sysroot --print sysroot)
  test $sysroot = $desired_sysroot

  sysroot=$(SYSROOT=$desired_sysroot ./target/debug/clippy-driver --print sysroot)
  test $sysroot = $desired_sysroot

  # Make sure this isn't set - clippy-driver should cope without it
  unset CARGO_MANIFEST_DIR

  # Run a lint and make sure it produces the expected output. It's also expected to exit with code 1
  # XXX How to match the clippy invocation in compile-test.rs?
  ! ./target/debug/clippy-driver -Dwarnings -Aunused -Zui-testing --emit metadata --crate-type bin tests/ui/cstring.rs 2> cstring.stderr
  diff <(sed -e 's,tests/ui,$DIR,' -e '/= help/d' cstring.stderr) tests/ui/cstring.stderr

  # TODO: CLIPPY_CONF_DIR / CARGO_MANIFEST_DIR
)
