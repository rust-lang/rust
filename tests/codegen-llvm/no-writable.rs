//! The tests here test that the `-Zno-writable` flag has the desired effect.
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Zno-writable
#![crate_type = "lib"]

// CHECK: @mutable_borrow(ptr noalias noundef align 4 dereferenceable(4) %_1)
#[no_mangle]
pub fn mutable_borrow(_: &mut i32) {}

// CHECK: @option_borrow_mut(ptr noalias noundef align 4 dereferenceable_or_null(4) %_x)
#[no_mangle]
pub fn option_borrow_mut(_x: Option<&mut i32>) {}
