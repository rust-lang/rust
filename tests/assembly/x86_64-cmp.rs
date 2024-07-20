//@ revisions: debug optim
//@ [debug] compile-flags: -C opt-level=0
//@ [optim] compile-flags: -C opt-level=3
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -C llvm-args=-x86-asm-syntax=intel
//@ only-x86_64
//@ ignore-sgx

#![feature(core_intrinsics)]

use std::intrinsics::three_way_compare;

#[no_mangle]
// CHECK-LABEL: signed_cmp:
pub fn signed_cmp(a: i16, b: i16) -> std::cmp::Ordering {
    // CHECK-DEBUG: cmp
    // CHECK-DEBUG: setg
    // CHECK-DEBUG: and
    // CHECK-DEBUG: cmp
    // CHECK-DEBUG: setl
    // CHECK-DEBUG: and
    // CHECK-DEBUG: sub

    // CHECK-OPTIM: xor
    // CHECK-OPTIM: cmp
    // CHECK-OPTIM: setne
    // CHECK-OPTIM: mov
    // CHECK-OPTIM: cmovge
    // CHECK-OPTIM: ret
    three_way_compare(a, b)
}

#[no_mangle]
// CHECK-LABEL: unsigned_cmp:
pub fn unsigned_cmp(a: u16, b: u16) -> std::cmp::Ordering {
    // CHECK-DEBUG: cmp
    // CHECK-DEBUG: seta
    // CHECK-DEBUG: and
    // CHECK-DEBUG: cmp
    // CHECK-DEBUG: setb
    // CHECK-DEBUG: and
    // CHECK-DEBUG: sub

    // CHECK-OPTIM: xor
    // CHECK-OPTIM: cmp
    // CHECK-OPTIM: setne
    // CHECK-OPTIM: mov
    // CHECK-OPTIM: cmovae
    // CHECK-OPTIM: ret
    three_way_compare(a, b)
}
