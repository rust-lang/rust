// This is test for more optimal Ord implementation for integers.
// See <https://github.com/rust-lang/rust/issues/63758> for more info.

// compile-flags: -C opt-level=3

#![crate_type = "lib"]

use std::cmp::Ordering;

// CHECK-LABEL: @cmp_signed
#[no_mangle]
pub fn cmp_signed(a: i64, b: i64) -> Ordering {
// CHECK: icmp slt
// CHECK: icmp ne
// CHECK: zext i1
// CHECK: select i1
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_unsigned
#[no_mangle]
pub fn cmp_unsigned(a: u32, b: u32) -> Ordering {
// CHECK: icmp ult
// CHECK: icmp ne
// CHECK: zext i1
// CHECK: select i1
    a.cmp(&b)
}
