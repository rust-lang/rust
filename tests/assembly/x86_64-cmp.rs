//@ revisions: DEBUG OPTIM
//@ [DEBUG] compile-flags: -C opt-level=0
//@ [OPTIM] compile-flags: -C opt-level=3
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

    // OPTIM: xor
    // OPTIM: cmp
    // OPTIM: setne
    // OPTIM: mov
    // OPTIM: cmovge
    // OPTIM: ret
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

    // OPTIM: xor
    // OPTIM: cmp
    // OPTIM: setne
    // OPTIM: mov
    // OPTIM: cmovae
    // OPTIM: ret
    three_way_compare(a, b)
}
