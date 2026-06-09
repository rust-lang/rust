// This is a regression test for https://github.com/rust-lang/rust/issues/139355

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::mem::MaybeUninit;

#[no_mangle]
pub fn create_uninit_array() -> [[MaybeUninit<u8>; 4]; 200] {
    // CHECK-LABEL: create_uninit_array
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void
    [[MaybeUninit::<u8>::uninit(); 4]; 200]
}
