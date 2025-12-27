//! This test checks that removing trailing zeroes from a `NonZero`,
//! then creating a new `NonZero` from the result does not panic.

//@ min-llvm-version: 21
//@ compile-flags: -O -Zmerge-functions=disabled
#![crate_type = "lib"]

use std::num::NonZero;

// CHECK-LABEL: @remove_trailing_zeros
#[no_mangle]
pub fn remove_trailing_zeros(x: NonZero<u8>) -> NonZero<u8> {
    // CHECK-NOT: unwrap_failed
    // CHECK-NOT: br
    // CHECK ret i8
    NonZero::new(x.get() >> x.trailing_zeros()).unwrap()
}
