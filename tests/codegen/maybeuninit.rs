//@ compile-flags: -Copt-level=3 -Cdebuginfo=0

// This is a regression test for https://github.com/rust-lang/rust/issues/139355 as well as
// regressions I introduced while implementing a solution.

#![crate_type = "lib"]

use std::mem::MaybeUninit;

// CHECK-LABEL: @create_small_uninit_array
#[no_mangle]
fn create_small_uninit_array() -> [MaybeUninit<u8>; 4] {
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret i32 undef
    [MaybeUninit::<u8>::uninit(); 4]
}

// CHECK-LABEL: @create_nested_uninit_array
#[no_mangle]
fn create_nested_uninit_array() -> [[MaybeUninit<u8>; 4]; 100] {
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void
    [[MaybeUninit::<u8>::uninit(); 4]; 100]
}

// CHECK-LABEL: @create_ptr
#[no_mangle]
fn create_ptr() -> MaybeUninit<&'static str> {
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret { ptr, i64 } undef
    MaybeUninit::uninit()
}
