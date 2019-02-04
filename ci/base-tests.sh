set -ex

echo "Running clippy base tests"

PATH=$PATH:./node_modules/.bin
if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  remark -f *.md > /dev/null
fi
# build clippy in debug mode and run tests
cargo build --features debugging
cargo test --features debugging
# for faster build, share target dir between subcrates
export CARGO_TARGET_DIR=`pwd`/target/
cd clippy_lints && cargo test && cd ..
cd rustc_tools_util && cargo test && cd ..
cd clippy_dev && cargo test && cd ..

# make sure clippy can be called via ./path/to/cargo-clippy
cd clippy_workspace_tests
../target/debug/cargo-clippy
cd ..

# Perform various checks for lint registration
./util/dev update_lints --check
cargo +nightly fmt --all -- --check

# make sure tests are formatted

# some lints are sensitive to formatting, exclude some files
tests_need_reformatting="false"
# switch to nightly
rustup override set nightly
# avoid loop spam and allow cmds with exit status != 0
set +ex

for file in `find tests -not -path "tests/ui/methods.rs" -not -path "tests/ui/format.rs" -not -path "tests/ui/formatting.rs" -not -path "tests/ui/empty_line_after_outer_attribute.rs" -not -path "tests/ui/double_parens.rs" -not -path "tests/ui/doc.rs" -not -path "tests/ui/unused_unit.rs" | grep "\.rs$"` ; do
  rustfmt ${file} --check
  if [ $? -ne 0 ]; then
    echo "${file} needs reformatting!"
    tests_need_reformatting="true"
  fi
done

set -ex # reset

if [ "${tests_need_reformatting}" == "true" ] ; then
    echo "Tests need reformatting!"
    exit 2
fi

# switch back to master
rustup override set master
