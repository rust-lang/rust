#!/bin/sh
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

    ${objdump} --disassemble-symbols=${func} --demangle \
      ${WORK_DIR}/enclave/target/x86_64-fortanix-unknown-sgx/debug/enclave > ${asm}
    ${filecheck} --input-file ${asm} ${checks}
}

build

check unw_getcontext unw_getcontext.checks
check "libunwind::Registers_x86_64::jumpto()" jumpto.checks
check "std::io::stdio::_print::h87f0c238421c45bc" print.checks
check rust_plus_one_global_asm rust_plus_one_global_asm.checks \
  || echo "warning: module level assembly currently not hardened"

check cc_plus_one_c cc_plus_one_c.checks
check cc_plus_one_c_asm cc_plus_one_c_asm.checks
check cc_plus_one_cxx cc_plus_one_cxx.checks
check cc_plus_one_cxx_asm cc_plus_one_cxx_asm.checks
check cc_plus_one_asm cc_plus_one_asm.checks \
  || echo "warning: the cc crate forwards assembly files to the CC compiler." \
           "Clang uses its own intergrated assembler, which does not include the LVI passes."

check cmake_plus_one_c cmake_plus_one_c.checks
check cmake_plus_one_c_asm cmake_plus_one_c_asm.checks
check cmake_plus_one_c_global_asm cmake_plus_one_c_global_asm.checks \
  || echo "warning: module level assembly currently not hardened"
check cmake_plus_one_cxx cmake_plus_one_cxx.checks
check cmake_plus_one_cxx_asm cmake_plus_one_cxx_asm.checks
check cmake_plus_one_cxx_global_asm cmake_plus_one_cxx_global_asm.checks \
  || echo "warning: module level assembly currently not hardened"
check cmake_plus_one_asm cmake_plus_one_asm.checks
