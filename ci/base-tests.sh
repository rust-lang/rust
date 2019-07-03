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
  export LD_LIBRARY_PATH=$(rustc --print sysroot)/lib

  # Check sysroot handling
  sysroot=$(./target/debug/clippy-driver --print sysroot)
  test $sysroot = $(rustc --print sysroot)

  sysroot=$(./target/debug/clippy-driver --sysroot /tmp --print sysroot)
  test $sysroot = /tmp

  sysroot=$(SYSROOT=/tmp ./target/debug/clippy-driver --print sysroot)
  test $sysroot = /tmp

  # Make sure this isn't set - clippy-driver should cope without it
  unset CARGO_MANIFEST_DIR

  # Run a lint and make sure it produces the expected output. It's also expected to exit with code 1
  # XXX How to match the clippy invocation in compile-test.rs?
  ! ./target/debug/clippy-driver -Dwarnings -Aunused -Zui-testing --emit metadata --crate-type bin tests/ui/cstring.rs 2> cstring.stderr
  diff <(sed -e 's,tests/ui,$DIR,' -e '/= help/d' cstring.stderr) tests/ui/cstring.stderr

  # TODO: CLIPPY_CONF_DIR / CARGO_MANIFEST_DIR
)

# make sure tests are formatted

# some lints are sensitive to formatting, exclude some files
tests_need_reformatting="false"
# switch to nightly
rustup override set nightly
# avoid loop spam and allow cmds with exit status != 0
set +ex

set -ex # reset

if [ "${tests_need_reformatting}" == "true" ] ; then
    echo "Tests need reformatting!"
    exit 2
fi

# switch back to master
rustup override set master
