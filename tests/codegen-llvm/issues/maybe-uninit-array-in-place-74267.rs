//! Regression test for <https://github.com/rust-lang/rust/issues/74267>.
//!
//! Returning `MaybeUninit::assume_init()` must build the array directly in the
//! return slot, without a temporary copy.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::mem::MaybeUninit;
use std::ptr;

// CHECK-LABEL: @init_array_in_place
#[no_mangle]
pub fn init_array_in_place() -> [String; 2] {
    // CHECK-NOT: alloca
    // CHECK-NOT: memcpy
    // CHECK: ret void
    let mut array: MaybeUninit<[String; 2]> = MaybeUninit::uninit();
    unsafe {
        let slots = &mut *array.as_mut_ptr();
        for slot in slots {
            ptr::write(slot, String::new());
        }
        array.assume_init()
    }
}
