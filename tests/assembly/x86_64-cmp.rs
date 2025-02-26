//@ revisions: DEBUG LLVM-PRE-20-OPTIM LLVM-20-OPTIM
//@ [DEBUG] compile-flags: -C opt-level=0
//@ [LLVM-PRE-20-OPTIM] compile-flags: -C opt-level=3
//@ [LLVM-PRE-20-OPTIM] max-llvm-major-version: 19
//@ [LLVM-20-OPTIM] compile-flags: -C opt-level=3
//@ [LLVM-20-OPTIM] min-llvm-version: 20
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -C llvm-args=-x86-asm-syntax=intel
//@ only-x86_64
//@ ignore-sgx

#![feature(core_intrinsics)]

use std::intrinsics::three_way_compare;

#[no_mangle]
// CHECK-LABEL: signed_cmp:
pub fn signed_cmp(a: i16, b: i16) -> std::cmp::Ordering {
    // DEBUG: cmp
    // DEBUG: setg
    // DEBUG: and
    // DEBUG: cmp
    // DEBUG: setl
    // DEBUG: and
    // DEBUG: sub

    // LLVM-PRE-20-OPTIM: xor
    // LLVM-PRE-20-OPTIM: cmp
    // LLVM-PRE-20-OPTIM: setne
    // LLVM-PRE-20-OPTIM: mov
    // LLVM-PRE-20-OPTIM: cmovge
    // LLVM-PRE-20-OPTIM: ret
    //
    // LLVM-20-OPTIM: cmp
    // LLVM-20-OPTIM: setl
    // LLVM-20-OPTIM: setg
    // LLVM-20-OPTIM: sub
    // LLVM-20-OPTIM: ret
    three_way_compare(a, b)
}

#[no_mangle]
// CHECK-LABEL: unsigned_cmp:
pub fn unsigned_cmp(a: u16, b: u16) -> std::cmp::Ordering {
    // DEBUG: cmp
    // DEBUG: seta
    // DEBUG: and
    // DEBUG: cmp
    // DEBUG: setb
    // DEBUG: and
    // DEBUG: sub

    // LLVM-PRE-20-OPTIM: xor
    // LLVM-PRE-20-OPTIM: cmp
    // LLVM-PRE-20-OPTIM: setne
    // LLVM-PRE-20-OPTIM: mov
    // LLVM-PRE-20-OPTIM: cmovae
    // LLVM-PRE-20-OPTIM: ret
    //
    // LLVM-20-OPTIM: cmp
    // LLVM-20-OPTIM: seta
    // LLVM-20-OPTIM: sbb
    // LLVM-20-OPTIM: ret
    three_way_compare(a, b)
}
