set -ex
PATH=$PATH:./node_modules/.bin
remark -f *.md > /dev/null
cargo build --features debugging
cargo test --features debugging
mkdir -p ~/rust/cargo/bin
cp target/debug/cargo-clippy ~/rust/cargo/bin/cargo-clippy
cp target/debug/clippy-driver ~/rust/cargo/bin/clippy-driver
rm ~/.cargo/bin/cargo-clippy
PATH=$PATH:~/rust/cargo/bin cargo clippy --all-targets --all-features -- --cap-lints warn -D clippy
cd clippy_workspace_tests && PATH=$PATH:~/rust/cargo/bin cargo clippy -- -D clippy && cd ..
cd clippy_workspace_tests/src && PATH=$PATH:~/rust/cargo/bin cargo clippy -- -D clippy && cd ../..
cd clippy_workspace_tests/subcrate && PATH=$PATH:~/rust/cargo/bin cargo clippy -- -D clippy && cd ../..
cd clippy_workspace_tests/subcrate/src && PATH=$PATH:~/rust/cargo/bin cargo clippy -- -D clippy && cd ../../..
PATH=$PATH:~/rust/cargo/bin cargo clippy --manifest-path=clippy_workspace_tests/Cargo.toml -- -D clippy
cd clippy_workspace_tests/subcrate && PATH=$PATH:~/rust/cargo/bin cargo clippy --manifest-path=../Cargo.toml -- -D clippy && cd ../..
set +x
