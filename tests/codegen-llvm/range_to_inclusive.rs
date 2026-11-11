//! Test that `RangeTo` and `RangeToInclusive` generate identical
//! (and optimal) code; #63646
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
#![crate_type = "lib"]

#[no_mangle]
// CHECK-LABEL: range_to(
pub fn range_to(a: i32, mut b: i32) -> i32 {
    // CHECK: %1 = and i32 %0, %a
    // CHECK-NEXT: ret i32 %1
    for _ in 0..65 {
        b &= a;
    }

    b
}

#[no_mangle]
// CHECK-LABEL: range_to_inclusive(
pub fn range_to_inclusive(a: i32, mut b: i32) -> i32 {
    // CHECK: %1 = and i32 %0, %a
    // CHECK-NEXT: ret i32 %1
    for _ in 0..=64 {
        b &= a;
    }

    b
}
