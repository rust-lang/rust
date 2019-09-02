// This is test for more optimal Ord implementation for integers.
// See <https://github.com/rust-lang/rust/pull/64082> for more info.

// compile-flags: -C opt-level=3

#![crate_type = "lib"]

use std::cmp::Ordering;

// CHECK-LABEL: @cmp_signed
#[no_mangle]
pub fn cmp_signed(a: i64, b: i64) -> Ordering {
// CHECK: icmp sgt
// CHECK: zext i1
// CHECK: icmp slt
// CHECK: zext i1
// CHECK: sub nsw
// CHECK-NOT: select
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_unsigned
#[no_mangle]
pub fn cmp_unsigned(a: u32, b: u32) -> Ordering {
// CHECK: icmp ugt
// CHECK: zext i1
// CHECK: icmp ult
// CHECK: zext i1
// CHECK: sub nsw
// CHECK-NOT: select
    a.cmp(&b)
}

// CHECK-LABEL: @cmp_signed_lt
#[no_mangle]
pub fn cmp_signed_lt(a: &i64, b: &i64) -> bool {
// CHECK: icmp slt
// CHECK-NOT: sub
// CHECK-NOT: select
    Ord::cmp(a, b) < Ordering::Equal
}

// CHECK-LABEL: @cmp_unsigned_lt
#[no_mangle]
pub fn cmp_unsigned_lt(a: &u32, b: &u32) -> bool {
// CHECK: icmp ult
// CHECK-NOT: sub
// CHECK-NOT: select
    Ord::cmp(a, b) < Ordering::Equal
}

// CHECK-LABEL: @cmp_signed_eq
#[no_mangle]
pub fn cmp_signed_eq(a: &i64, b: &i64) -> bool {
// CHECK: icmp eq
// CHECK-NOT: sub
// CHECK-NOT: select
    Ord::cmp(a, b) == Ordering::Equal
}

// CHECK-LABEL: @cmp_unsigned_eq
#[no_mangle]
pub fn cmp_unsigned_eq(a: &u32, b: &u32) -> bool {
// CHECK: icmp eq
// CHECK-NOT: sub
// CHECK-NOT: select
    Ord::cmp(a, b) == Ordering::Equal
}
