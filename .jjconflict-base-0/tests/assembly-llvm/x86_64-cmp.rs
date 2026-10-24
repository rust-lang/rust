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
    // OPTIM: cmp
    // CHECK: setl
    // CHECK: setg
    // CHECK: sub
    // CHECK: ret
    three_way_compare(a, b)
}

#[no_mangle]
// CHECK-LABEL: unsigned_cmp:
pub fn unsigned_cmp(a: u16, b: u16) -> std::cmp::Ordering {
    // DEBUG: sub
    // OPTIM: cmp
    // CHECK: seta
    // CHECK: sbb
    // CHECK: ret
    three_way_compare(a, b)
}
