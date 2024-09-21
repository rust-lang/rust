//@ compile-flags: -O -Zmerge-functions=disabled
#![crate_type = "lib"]

extern crate core;
use core::cmp::Ordering;
use core::num::NonZero;
use core::ptr::NonNull;

// CHECK-LABEL: @non_zero_eq
#[no_mangle]
pub fn non_zero_eq(l: Option<NonZero<u32>>, r: Option<NonZero<u32>>) -> bool {
    // CHECK: start:
    // CHECK-NEXT: icmp eq i32
    // CHECK-NEXT: ret i1
    l == r
}

// CHECK-LABEL: @non_zero_signed_eq
#[no_mangle]
pub fn non_zero_signed_eq(l: Option<NonZero<i64>>, r: Option<NonZero<i64>>) -> bool {
    // CHECK: start:
    // CHECK-NEXT: icmp eq i64
    // CHECK-NEXT: ret i1
    l == r
}

// CHECK-LABEL: @non_null_eq
#[no_mangle]
pub fn non_null_eq(l: Option<NonNull<u8>>, r: Option<NonNull<u8>>) -> bool {
    // CHECK: start:
    // CHECK-NEXT: icmp eq ptr
    // CHECK-NEXT: ret i1
    l == r
}

// CHECK-LABEL: @ordering_eq
#[no_mangle]
pub fn ordering_eq(l: Option<Ordering>, r: Option<Ordering>) -> bool {
    // CHECK: start:
    // CHECK-NEXT: icmp eq i8
    // CHECK-NEXT: ret i1
    l == r
}

#[derive(PartialEq)]
pub enum EnumWithNiche {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
}

// CHECK-LABEL: @niche_eq
#[no_mangle]
pub fn niche_eq(l: Option<EnumWithNiche>, r: Option<EnumWithNiche>) -> bool {
    // CHECK: start:
    // CHECK-NEXT: icmp eq i8
    // CHECK-NEXT: ret i1
    l == r
}

// FIXME: This should work too
// // FIXME-CHECK-LABEL: @bool_eq
// #[no_mangle]
// pub fn bool_eq(l: Option<bool>, r: Option<bool>) -> bool {
//     // FIXME-CHECK: start:
//     // FIXME-CHECK-NEXT: icmp eq i8
//     // FIXME-CHECK-NEXT: ret i1
//     l == r
// }
