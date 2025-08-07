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
    // DEBUG: sub
    // DEBUG: setl
    // DEBUG: setg
    // DEBUG: sub
    // DEBUG: ret
    //
    // OPTIM: cmp
    // OPTIM: setl
    // OPTIM: setg
    // OPTIM: sub
    // OPTIM: ret
    three_way_compare(a, b)
}

#[no_mangle]
// CHECK-LABEL: unsigned_cmp:
pub fn unsigned_cmp(a: u16, b: u16) -> std::cmp::Ordering {
    // DEBUG: sub
    // DEBUG: seta
    // DEBUG: sbb
    // DEBUG: ret
    //
    // OPTIM: cmp
    // OPTIM: seta
    // OPTIM: sbb
    // OPTIM: ret
    three_way_compare(a, b)
}
