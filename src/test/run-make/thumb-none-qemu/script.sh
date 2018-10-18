set -exuo pipefail

CRATE=example

env | sort
mkdir -p $WORK_DIR
pushd $WORK_DIR
    rm -rf $CRATE || echo OK
    cp -a $HERE/example .
    pushd $CRATE
        env RUSTFLAGS="-C linker=arm-none-eabi-ld -C link-arg=-Tlink.x" \
            $CARGO run --target $TARGET           | grep "x = 42"
        env RUSTFLAGS="-C linker=arm-none-eabi-ld -C link-arg=-Tlink.x" \
            $CARGO run --target $TARGET --release | grep "x = 42"
    popd
popd
