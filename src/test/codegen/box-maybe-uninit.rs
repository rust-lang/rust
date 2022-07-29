// compile-flags: -O
#![crate_type="lib"]

use std::mem::MaybeUninit;

// Boxing a `MaybeUninit` value should not copy junk from the stack
#[no_mangle]
pub fn box_uninitialized() -> Box<MaybeUninit<usize>> {
    // CHECK-LABEL: @box_uninitialized
    // CHECK-NOT: store
    // CHECK-NOT: alloca
    // CHECK-NOT: memcpy
    // CHECK-NOT: memset
    Box::new(MaybeUninit::uninit())
}

// FIXME: add a test for a bigger box. Currently broken, see
// https://github.com/rust-lang/rust/issues/58201.
