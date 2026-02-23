//! This test checks that removing trailing zeroes from a `NonZero`,
//! then creating a new `NonZero` from the result does not panic.

//@ min-llvm-version: 21
//@ compile-flags: -O -Zmerge-functions=disabled
#![crate_type = "lib"]

use std::num::NonZero;

// CHECK-LABEL: @remove_trailing_zeros
#[no_mangle]
pub fn remove_trailing_zeros(x: NonZero<u8>) -> NonZero<u8> {
    // CHECK: %[[TRAILING:[a-z0-9_-]+]] = {{.*}} call {{.*}} i8 @llvm.cttz.i8(i8 %x, i1 true)
    // CHECK-NEXT: %[[RET:[a-z0-9_-]+]] = lshr exact i8 %x, %[[TRAILING]]
    // CHECK-NEXT: ret i8 %[[RET]]
    NonZero::new(x.get() >> x.trailing_zeros()).unwrap()
}
