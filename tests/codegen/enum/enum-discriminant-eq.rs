//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled

// The `derive(PartialEq)` on enums with field-less variants compares discriminants,
// so make sure we emit that in some reasonable way.

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(ascii_char)]

use std::ascii::Char as AC;
use std::cmp::Ordering;
use std::intrinsics::discriminant_value;
use std::num::NonZero;

#[unsafe(no_mangle)]
pub fn opt_bool_eq(a: Option<bool>, b: Option<bool>) -> bool {
    // CHECK-LABEL: @opt_bool_eq(
    // CHECK: %[[A:.+]] = icmp ne i8 %a, 2
    // CHECK: %[[B:.+]] = icmp eq i8 %b, 2
    // CHECK: %[[R:.+]] = xor i1 %[[A]], %[[B]]
    // CHECK: ret i1 %[[R]]

    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn opt_ord_eq(a: Option<Ordering>, b: Option<Ordering>) -> bool {
    // CHECK-LABEL: @opt_ord_eq(
    // CHECK: %[[A:.+]] = icmp ne i8 %a, 2
    // CHECK: %[[B:.+]] = icmp eq i8 %b, 2
    // CHECK: %[[R:.+]] = xor i1 %[[A]], %[[B]]
    // CHECK: ret i1 %[[R]]

    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn opt_nz32_eq(a: Option<NonZero<u32>>, b: Option<NonZero<u32>>) -> bool {
    // CHECK-LABEL: @opt_nz32_eq(
    // CHECK: %[[A:.+]] = icmp ne i32 %a, 0
    // CHECK: %[[B:.+]] = icmp eq i32 %b, 0
    // CHECK: %[[R:.+]] = xor i1 %[[A]], %[[B]]
    // CHECK: ret i1 %[[R]]

    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn opt_ac_eq(a: Option<AC>, b: Option<AC>) -> bool {
    // CHECK-LABEL: @opt_ac_eq(
    // CHECK: %[[A:.+]] = icmp ne i8 %a, -128
    // CHECK: %[[B:.+]] = icmp eq i8 %b, -128
    // CHECK: %[[R:.+]] = xor i1 %[[A]], %[[B]]
    // CHECK: ret i1 %[[R]]

    discriminant_value(&a) == discriminant_value(&b)
}

pub enum Mid<T> {
    Before,
    Thing(T),
    After,
}

#[unsafe(no_mangle)]
pub fn mid_bool_eq(a: Mid<bool>, b: Mid<bool>) -> bool {
    // CHECK-LABEL: @mid_bool_eq(
    // CHECK: %[[AS:.+]] = add nsw i8 %a, -2
    // CHECK: %[[AT:.+]] = icmp ult i8 %[[AS]], 3
    // CHECK: %[[AD:.+]] = select i1 %[[AT]], i8 %[[AS]], i8 1
    // CHECK: %[[BS:.+]] = add nsw i8 %b, -2
    // CHECK: %[[BT:.+]] = icmp ult i8 %[[BS]], 3
    // CHECK: %[[BD:.+]] = select i1 %[[BT]], i8 %[[BS]], i8 1
    // CHECK: %[[R:.+]] = icmp eq i8 %[[AD]], %[[BD]]
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn mid_ord_eq(a: Mid<Ordering>, b: Mid<Ordering>) -> bool {
    // CHECK-LABEL: @mid_ord_eq(
    // CHECK: %[[AS:.+]] = add nsw i8 %a, -2
    // CHECK: %[[AT:.+]] = icmp ult i8 %[[AS]], 3
    // CHECK: %[[AD:.+]] = select i1 %[[AT]], i8 %[[AS]], i8 1
    // CHECK: %[[BS:.+]] = add nsw i8 %b, -2
    // CHECK: %[[BT:.+]] = icmp ult i8 %[[BS]], 3
    // CHECK: %[[BD:.+]] = select i1 %[[BT]], i8 %[[BS]], i8 1
    // CHECK: %[[R:.+]] = icmp eq i8 %[[AD]], %[[BD]]
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn mid_nz32_eq(a: Mid<NonZero<u32>>, b: Mid<NonZero<u32>>) -> bool {
    // CHECK-LABEL: @mid_nz32_eq(
    // CHECK: %[[R:.+]] = icmp eq i32 %a.0, %b.0
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn mid_ac_eq(a: Mid<AC>, b: Mid<AC>) -> bool {
    // CHECK-LABEL: @mid_ac_eq(
    // CHECK: %[[AS:.+]] = xor i8 %a, -128
    // CHECK: %[[AT:.+]] = icmp ult i8 %[[AS]], 3
    // CHECK: %[[AD:.+]] = select i1 %[[AT]], i8 %[[AS]], i8 1
    // CHECK: %[[BS:.+]] = xor i8 %b, -128
    // CHECK: %[[BT:.+]] = icmp ult i8 %[[BS]], 3
    // CHECK: %[[BD:.+]] = select i1 %[[BT]], i8 %[[BS]], i8 1
    // CHECK: %[[R:.+]] = icmp eq i8 %[[AD]], %[[BD]]
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}
