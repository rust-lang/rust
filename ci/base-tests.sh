set -ex

echo "Running clippy base tests"

PATH=$PATH:./node_modules/.bin
remark -f *.md > /dev/null
# build clippy in debug mode and run tests
cargo build --features debugging
cargo test --features debugging
mkdir -p ~/rust/cargo/bin
cp target/debug/cargo-clippy ~/rust/cargo/bin/cargo-clippy
cp target/debug/clippy-driver ~/rust/cargo/bin/clippy-driver
rm ~/.cargo/bin/cargo-clippy
# run clippy on its own codebase...
PATH=$PATH:~/rust/cargo/bin cargo clippy --all-targets --all-features -- -D clippy
# ... and some test directories
cd clippy_workspace_tests && PATH=$PATH:~/rust/cargo/bin cargo clippy -- -D clippy && cd ..
cd clippy_workspace_tests/src && PATH=$PATH:~/rust/cargo/bin cargo clippy -- -D clippy && cd ../..
cd clippy_workspace_tests/subcrate && PATH=$PATH:~/rust/cargo/bin cargo clippy -- -D clippy && cd ../..
cd clippy_workspace_tests/subcrate/src && PATH=$PATH:~/rust/cargo/bin cargo clippy -- -D clippy && cd ../../..
# test --manifest-path
PATH=$PATH:~/rust/cargo/bin cargo clippy --manifest-path=clippy_workspace_tests/Cargo.toml -- -D clippy
cd clippy_workspace_tests/subcrate && PATH=$PATH:~/rust/cargo/bin cargo clippy --manifest-path=../Cargo.toml -- -D clippy && cd ../..
set +x
