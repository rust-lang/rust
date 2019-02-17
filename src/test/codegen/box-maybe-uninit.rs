// compile-flags: -O
#![crate_type="lib"]
#![feature(maybe_uninit)]

use std::mem::MaybeUninit;

// Boxing a `MaybeUninit` value should not copy junk from the stack
#[no_mangle]
pub fn box_uninitialized() -> Box<MaybeUninit<usize>> {
    // CHECK-LABEL: @box_uninitialized
    // CHECK-NOT: store
    // CHECK-NOT: alloca
    // CHECK-NOT: memcpy
    // CHECK-NOT: memset
    Box::new(MaybeUninit::uninitialized())
}
