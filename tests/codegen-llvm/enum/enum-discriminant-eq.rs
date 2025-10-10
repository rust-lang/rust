//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
//@ only-64bit
//@ revisions: LLVM20 LLVM21
//@ [LLVM21] min-llvm-version: 21
//@ [LLVM20] max-llvm-major-version: 20

// The `derive(PartialEq)` on enums with field-less variants compares discriminants,
// so make sure we emit that in some reasonable way.

#![crate_type = "lib"]
#![feature(ascii_char)]
#![feature(core_intrinsics)]
#![feature(repr128)]

use std::ascii::Char as AC;
use std::cmp::Ordering;
use std::intrinsics::discriminant_value;
use std::num::NonZero;

// A type that's bigger than `isize`, unlike the usual cases that have small tags.
#[repr(u128)]
pub enum Giant {
    Two = 2,
    Three = 3,
    Four = 4,
}

#[unsafe(no_mangle)]
pub fn opt_bool_eq_discr(a: Option<bool>, b: Option<bool>) -> bool {
    // CHECK-LABEL: @opt_bool_eq_discr(
    // CHECK: %[[A:.+]] = icmp ne i8 %a, 2
    // CHECK: %[[B:.+]] = icmp eq i8 %b, 2
    // CHECK: %[[R:.+]] = xor i1 %[[A]], %[[B]]
    // CHECK: ret i1 %[[R]]

    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn opt_ord_eq_discr(a: Option<Ordering>, b: Option<Ordering>) -> bool {
    // CHECK-LABEL: @opt_ord_eq_discr(
    // CHECK: %[[A:.+]] = icmp ne i8 %a, 2
    // CHECK: %[[B:.+]] = icmp eq i8 %b, 2
    // CHECK: %[[R:.+]] = xor i1 %[[A]], %[[B]]
    // CHECK: ret i1 %[[R]]

    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn opt_nz32_eq_discr(a: Option<NonZero<u32>>, b: Option<NonZero<u32>>) -> bool {
    // CHECK-LABEL: @opt_nz32_eq_discr(
    // CHECK: %[[A:.+]] = icmp ne i32 %a, 0
    // CHECK: %[[B:.+]] = icmp eq i32 %b, 0
    // CHECK: %[[R:.+]] = xor i1 %[[A]], %[[B]]
    // CHECK: ret i1 %[[R]]

    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn opt_ac_eq_discr(a: Option<AC>, b: Option<AC>) -> bool {
    // CHECK-LABEL: @opt_ac_eq_discr(
    // CHECK: %[[A:.+]] = icmp ne i8 %a, -128
    // CHECK: %[[B:.+]] = icmp eq i8 %b, -128
    // CHECK: %[[R:.+]] = xor i1 %[[A]], %[[B]]
    // CHECK: ret i1 %[[R]]

    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn opt_giant_eq_discr(a: Option<Giant>, b: Option<Giant>) -> bool {
    // CHECK-LABEL: @opt_giant_eq_discr(
    // CHECK: %[[A:.+]] = icmp ne i128 %a, 1
    // CHECK: %[[B:.+]] = icmp eq i128 %b, 1
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
pub fn mid_bool_eq_discr(a: Mid<bool>, b: Mid<bool>) -> bool {
    // CHECK-LABEL: @mid_bool_eq_discr(

    // CHECK: %[[A_NOT_HOLE:.+]] = icmp ne i8 %a, 3
    // CHECK: tail call void @llvm.assume(i1 %[[A_NOT_HOLE]])
    // LLVM20: %[[A_REL_DISCR:.+]] = add nsw i8 %a, -2
    // CHECK: %[[A_IS_NICHE:.+]] = icmp samesign ugt i8 %a, 1
    // LLVM20: %[[A_DISCR:.+]] = select i1 %[[A_IS_NICHE]], i8 %[[A_REL_DISCR]], i8 1

    // CHECK: %[[B_NOT_HOLE:.+]] = icmp ne i8 %b, 3
    // CHECK: tail call void @llvm.assume(i1 %[[B_NOT_HOLE]])
    // LLVM20: %[[B_REL_DISCR:.+]] = add nsw i8 %b, -2
    // CHECK: %[[B_IS_NICHE:.+]] = icmp samesign ugt i8 %b, 1
    // LLVM20: %[[B_DISCR:.+]] = select i1 %[[B_IS_NICHE]], i8 %[[B_REL_DISCR]], i8 1

    // LLVM21: %[[A_MOD_DISCR:.+]] = select i1 %[[A_IS_NICHE]], i8 %a, i8 3
    // LLVM21: %[[B_MOD_DISCR:.+]] = select i1 %[[B_IS_NICHE]], i8 %b, i8 3

    // LLVM20: %[[R:.+]] = icmp eq i8 %[[A_DISCR]], %[[B_DISCR]]
    // LLVM21: %[[R:.+]] = icmp eq i8 %[[A_MOD_DISCR]], %[[B_MOD_DISCR]]
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn mid_ord_eq_discr(a: Mid<Ordering>, b: Mid<Ordering>) -> bool {
    // CHECK-LABEL: @mid_ord_eq_discr(

    // CHECK: %[[A_NOT_HOLE:.+]] = icmp ne i8 %a, 3
    // CHECK: tail call void @llvm.assume(i1 %[[A_NOT_HOLE]])
    // LLVM20: %[[A_REL_DISCR:.+]] = add nsw i8 %a, -2
    // CHECK: %[[A_IS_NICHE:.+]] = icmp sgt i8 %a, 1
    // LLVM20: %[[A_DISCR:.+]] = select i1 %[[A_IS_NICHE]], i8 %[[A_REL_DISCR]], i8 1

    // CHECK: %[[B_NOT_HOLE:.+]] = icmp ne i8 %b, 3
    // CHECK: tail call void @llvm.assume(i1 %[[B_NOT_HOLE]])
    // LLVM20: %[[B_REL_DISCR:.+]] = add nsw i8 %b, -2
    // CHECK: %[[B_IS_NICHE:.+]] = icmp sgt i8 %b, 1
    // LLVM20: %[[B_DISCR:.+]] = select i1 %[[B_IS_NICHE]], i8 %[[B_REL_DISCR]], i8 1

    // LLVM21: %[[A_MOD_DISCR:.+]] = select i1 %[[A_IS_NICHE]], i8 %a, i8 3
    // LLVM21: %[[B_MOD_DISCR:.+]] = select i1 %[[B_IS_NICHE]], i8 %b, i8 3

    // LLVM20: %[[R:.+]] = icmp eq i8 %[[A_DISCR]], %[[B_DISCR]]
    // LLVM21: %[[R:.+]] = icmp eq i8 %[[A_MOD_DISCR]], %[[B_MOD_DISCR]]
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn mid_nz32_eq_discr(a: Mid<NonZero<u32>>, b: Mid<NonZero<u32>>) -> bool {
    // CHECK-LABEL: @mid_nz32_eq_discr(
    // CHECK: %[[R:.+]] = icmp eq i32 %a.0, %b.0
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}

#[unsafe(no_mangle)]
pub fn mid_ac_eq_discr(a: Mid<AC>, b: Mid<AC>) -> bool {
    // CHECK-LABEL: @mid_ac_eq_discr(

    // CHECK: %[[A_NOT_HOLE:.+]] = icmp ne i8 %a, -127
    // CHECK: tail call void @llvm.assume(i1 %[[A_NOT_HOLE]])
    // LLVM20: %[[A_REL_DISCR:.+]] = xor i8 %a, -128
    // CHECK: %[[A_IS_NICHE:.+]] = icmp slt i8 %a, 0
    // LLVM20: %[[A_DISCR:.+]] = select i1 %[[A_IS_NICHE]], i8 %[[A_REL_DISCR]], i8 1

    // CHECK: %[[B_NOT_HOLE:.+]] = icmp ne i8 %b, -127
    // CHECK: tail call void @llvm.assume(i1 %[[B_NOT_HOLE]])
    // LLVM20: %[[B_REL_DISCR:.+]] = xor i8 %b, -128
    // CHECK: %[[B_IS_NICHE:.+]] = icmp slt i8 %b, 0
    // LLVM20: %[[B_DISCR:.+]] = select i1 %[[B_IS_NICHE]], i8 %[[B_REL_DISCR]], i8 1

    // LLVM21: %[[A_DISCR:.+]] = select i1 %[[A_IS_NICHE]], i8 %a, i8 -127
    // LLVM21: %[[B_DISCR:.+]] = select i1 %[[B_IS_NICHE]], i8 %b, i8 -127

    // CHECK: %[[R:.+]] = icmp eq i8 %[[A_DISCR]], %[[B_DISCR]]
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}

// FIXME: This should be improved once our LLVM fork picks up the fix for
// <https://github.com/llvm/llvm-project/issues/134024>
#[unsafe(no_mangle)]
pub fn mid_giant_eq_discr(a: Mid<Giant>, b: Mid<Giant>) -> bool {
    // CHECK-LABEL: @mid_giant_eq_discr(

    // CHECK: %[[A_NOT_HOLE:.+]] = icmp ne i128 %a, 6
    // CHECK: tail call void @llvm.assume(i1 %[[A_NOT_HOLE]])
    // CHECK: %[[A_TRUNC:.+]] = trunc nuw nsw i128 %a to i64
    // LLVM20: %[[A_REL_DISCR:.+]] = add nsw i64 %[[A_TRUNC]], -5
    // CHECK: %[[A_IS_NICHE:.+]] = icmp samesign ugt i128 %a, 4
    // LLVM20: %[[A_DISCR:.+]] = select i1 %[[A_IS_NICHE]], i64 %[[A_REL_DISCR]], i64 1

    // CHECK: %[[B_NOT_HOLE:.+]] = icmp ne i128 %b, 6
    // CHECK: tail call void @llvm.assume(i1 %[[B_NOT_HOLE]])
    // CHECK: %[[B_TRUNC:.+]] = trunc nuw nsw i128 %b to i64
    // LLVM20: %[[B_REL_DISCR:.+]] = add nsw i64 %[[B_TRUNC]], -5
    // CHECK: %[[B_IS_NICHE:.+]] = icmp samesign ugt i128 %b, 4
    // LLVM20: %[[B_DISCR:.+]] = select i1 %[[B_IS_NICHE]], i64 %[[B_REL_DISCR]], i64 1

    // LLVM21: %[[A_MODIFIED_TAG:.+]] = select i1 %[[A_IS_NICHE]], i64 %[[A_TRUNC]], i64 6
    // LLVM21: %[[B_MODIFIED_TAG:.+]] = select i1 %[[B_IS_NICHE]], i64 %[[B_TRUNC]], i64 6
    // LLVM21: %[[R:.+]] = icmp eq i64 %[[A_MODIFIED_TAG]], %[[B_MODIFIED_TAG]]

    // LLVM20: %[[R:.+]] = icmp eq i64 %[[A_DISCR]], %[[B_DISCR]]
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == discriminant_value(&b)
}

// In niche-encoded enums, testing for the untagged variant should optimize to a
// straight-forward comparison looking for the natural range of the payload value.

#[unsafe(no_mangle)]
pub fn mid_bool_is_thing(a: Mid<bool>) -> bool {
    // CHECK-LABEL: @mid_bool_is_thing(
    // CHECK: %[[R:.+]] = icmp samesign ult i8 %a, 2
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == 1
}

#[unsafe(no_mangle)]
pub fn mid_ord_is_thing(a: Mid<Ordering>) -> bool {
    // CHECK-LABEL: @mid_ord_is_thing(
    // CHECK: %[[R:.+]] = icmp slt i8 %a, 2
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == 1
}

#[unsafe(no_mangle)]
pub fn mid_nz32_is_thing(a: Mid<NonZero<u32>>) -> bool {
    // CHECK-LABEL: @mid_nz32_is_thing(
    // CHECK: %[[R:.+]] = icmp eq i32 %a.0, 1
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == 1
}

#[unsafe(no_mangle)]
pub fn mid_ac_is_thing(a: Mid<AC>) -> bool {
    // CHECK-LABEL: @mid_ac_is_thing(
    // CHECK: %[[R:.+]] = icmp sgt i8 %a, -1
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == 1
}

#[unsafe(no_mangle)]
pub fn mid_giant_is_thing(a: Mid<Giant>) -> bool {
    // CHECK-LABEL: @mid_giant_is_thing(
    // CHECK: %[[R:.+]] = icmp samesign ult i128 %a, 5
    // CHECK: ret i1 %[[R]]
    discriminant_value(&a) == 1
}
