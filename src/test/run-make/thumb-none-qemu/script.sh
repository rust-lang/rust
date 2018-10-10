set -exuo pipefail

CRATE=cortex-m-rt
CRATE_URL=https://github.com/rust-embedded/cortex-m-rt
CRATE_SHA1=62972c8a89ff54b76f9ef0d600c1fcf7a233aabd

env | sort
mkdir -p $WORK_DIR
pushd $WORK_DIR
    rm -rf $CRATE || echo OK
    bash -x $HERE/../git_clone_sha1.sh $CRATE $CRATE_URL $CRATE_SHA1
    pushd $CRATE
        $CARGO run --target $TARGET --example qemu           | grep "x = 42"
        $CARGO run --target $TARGET --example qemu --release | grep "x = 42"
    popd
popd