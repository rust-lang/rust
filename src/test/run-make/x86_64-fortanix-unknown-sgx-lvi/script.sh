set -exuo pipefail

function build {
    CRATE=enclave

    mkdir -p $WORK_DIR
    pushd $WORK_DIR
        rm -rf $CRATE
        cp -a $TEST_DIR/enclave .
        pushd $CRATE
            echo ${WORK_DIR}
            # HACK(eddyb) sets `RUSTC_BOOTSTRAP=1` so Cargo can accept nightly features.
            # These come from the top-level Rust workspace, that this crate is not a
            # member of, but Cargo tries to load the workspace `Cargo.toml` anyway.
            env RUSTC_BOOTSTRAP=1
                cargo -v run --target $TARGET
        popd
    popd
}

function check {
    local func=$1
    local checks="${TEST_DIR}/$2"
    local asm=$(mktemp)
    local objdump="${BUILD_DIR}/x86_64-unknown-linux-gnu/llvm/build/bin/llvm-objdump"
    local filecheck="${BUILD_DIR}/x86_64-unknown-linux-gnu/llvm/build/bin/FileCheck"
    
    ${objdump} --disassemble-symbols=${func} --demangle ${WORK_DIR}/enclave/target/x86_64-fortanix-unknown-sgx/debug/enclave > ${asm}
    ${filecheck} --input-file ${asm} ${checks}
}

build

#TODO: re-enable check when newly compiled libunwind is used
#check unw_getcontext unw_getcontext.checks

#TODO: re-enable check when newly compiled libunwind is used
#check "libunwind::Registers_x86_64::jumpto()" jumpto.checks

check "std::io::stdio::_print::h87f0c238421c45bc" print.checks
check cc_plus_one_c cc_plus_one_c.checks
