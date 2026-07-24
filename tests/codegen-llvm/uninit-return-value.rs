// Regression test for https://github.com/rust-lang/rust/issues/159815

//@ compile-flags: -Copt-level=2

#![crate_type = "lib"]

use std::mem::MaybeUninit;

// CHECK-LABEL: @f
#[no_mangle]
pub fn f() -> MaybeUninit<*const ()> {
    // CHECK: start:
    // CHECK-NEXT: ret ptr undef
    MaybeUninit::uninit()
}
