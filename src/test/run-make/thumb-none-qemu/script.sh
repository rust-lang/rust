set -exuo pipefail

CRATE=example

env | sort
mkdir -p $WORK_DIR
pushd $WORK_DIR
    rm -rf $CRATE || echo OK
    cp -a $HERE/example .
    pushd $CRATE
        $CARGO run --target $TARGET           | grep "x = 42"
        $CARGO run --target $TARGET --release | grep "x = 42"
    popd
popd
