//! The tests here test that the `-Zllvm-writable` flag and
//! the `#[rustc_no_writable]` attribute have the desired effect.
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Zllvm-writable
#![crate_type = "lib"]
#![feature(rustc_attrs, unsafe_pinned)]

// CHECK: @mutable_borrow(ptr noalias noundef writable align 4 dereferenceable(4) %_1)
#[no_mangle]
pub fn mutable_borrow(_: &mut i32) {}

// CHECK: @mutable_unsafe_borrow(ptr noalias noundef writable align 2 dereferenceable(2) %_1)
#[no_mangle]
pub fn mutable_unsafe_borrow(_: &mut std::cell::UnsafeCell<i16>) {}

// CHECK: @option_borrow_mut(ptr noalias noundef writable align 4 dereferenceable_or_null(4) %_1)
#[no_mangle]
pub fn option_borrow_mut(_: Option<&mut i32>) {}

// CHECK: @box_moved(ptr noalias noundef nonnull align 4 %0)
#[no_mangle]
pub fn box_moved(_: Box<i32>) {}

// CHECK: @unsafe_pinned_borrow_mut(ptr noundef nonnull align 4 %_1)
#[no_mangle]
pub fn unsafe_pinned_borrow_mut(_: &mut std::pin::UnsafePinned<i32>) {}

// CHECK: @mutable_borrow_no_writable(ptr noalias noundef align 4 dereferenceable(4) %_1)
#[no_mangle]
#[rustc_no_writable]
pub fn mutable_borrow_no_writable(_: &mut i32) {}
