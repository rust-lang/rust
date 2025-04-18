//@ compile-flags: -Copt-level=1 -C no-prepopulate-passes
//@ only-64bit (because these discriminants are isize)

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// Depending on the relative ordering of the variants, either
//
// 1. We want to `llvm.assume` to make it clear that the side with the niched
//    tags can't actually have a value corresponding to the untagged one, or
//
// 2. The untagged variant would actually be on the side with the values,
//    where it's critical that we *don't* assume since that could be one of
//    the natural values, and thus we'd introduce UB.
//
// so these tests are particularly about *not* having assumes in the latter case.

// See also `enum-discriminant-eq.rs`, which has payload-in-the-middle tests.
// (That's not actually different in how it's detected during codegen compared
//  to the cases here, but it's more relevant to how tests get optimized.)

use std::cmp::Ordering;
use std::intrinsics::discriminant_value;

pub enum PayloadFirst<T> {
    Payload(T),
    After1,
    After2,
}

pub enum PayloadLast<T> {
    Before1,
    Before2,
    Payload(T),
}

// For a bool payload, the niches are 2 and 3.
// - with the payload first, the payload variant equivalent is 1, which is a valid value.
// - with the payload last, the payload variant equivalent is 4, which we assume away.

#[unsafe(no_mangle)]
pub fn payload_first_bool(a: PayloadFirst<bool>) -> isize {
    // CHECK-LABEL: @payload_first_bool(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = zext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp uge i8 %a, 2
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 1
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], -1
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

#[unsafe(no_mangle)]
pub fn payload_last_bool(a: PayloadLast<bool>) -> isize {
    // CHECK-LABEL: @payload_last_bool(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = zext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp uge i8 %a, 2
    // CHECK-NEXT: %[[ASSUME:.+]] = icmp ne i64 %0, 4
    // CHECK-NEXT: call void @llvm.assume(i1 %[[ASSUME]])
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 4
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], -2
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

// For a 7/8/9 payload, niches are 5 and 6, *before* the payload values.
// - with the payload first, the payload variant equivalent is 4, which we assume away.
// - with the payload last, the payload variant equivalent is 7, which is a valid value.

pub enum SevenEightNine {
    Seven = 7,
    Eight = 8,
    Nine = 9,
}

#[unsafe(no_mangle)]
pub fn payload_first_789(a: PayloadFirst<SevenEightNine>) -> isize {
    // CHECK-LABEL: @payload_first_789(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = zext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp ule i8 %a, 6
    // CHECK-NEXT: %[[ASSUME:.+]] = icmp ne i64 %0, 4
    // CHECK-NEXT: call void @llvm.assume(i1 %[[ASSUME]])
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 4
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], -4
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

#[unsafe(no_mangle)]
pub fn payload_last_789(a: PayloadLast<SevenEightNine>) -> isize {
    // CHECK-LABEL: @payload_last_789(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = zext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp ule i8 %a, 6
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 7
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], -5
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

// For a 2/3/4 payload, the zero niche gets prioritized so the niches are 0 and 1.
// - with the payload first, the payload variant equivalent wraps to isize::MAX,
//   which is actually on the value side again, so we don't assume it.
// - with the payload last, the payload variant equivalent is 2, which is a valid value.
//   (It also happens to have the tag equal to the discriminant, no adjustment needed.)

pub enum TwoThreeFour {
    Two = 2,
    Three = 3,
    Four = 4,
}

#[unsafe(no_mangle)]
pub fn payload_first_234(a: PayloadFirst<TwoThreeFour>) -> isize {
    // CHECK-LABEL: @payload_first_234(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = zext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp ule i8 %a, 1
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 -1
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], 1
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

#[unsafe(no_mangle)]
pub fn payload_last_234(a: PayloadLast<TwoThreeFour>) -> isize {
    // CHECK-LABEL: @payload_last_234(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = zext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp ule i8 %a, 1
    // CHECK-NEXT: %[[DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 2
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

// For an Ordering payload, the niches are 2 and 3 -- like `bool` but signed.

#[unsafe(no_mangle)]
pub fn payload_first_ordering(a: PayloadFirst<Ordering>) -> isize {
    // CHECK-LABEL: @payload_first_ordering(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = sext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp sge i8 %a, 2
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 1
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], -1
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

#[unsafe(no_mangle)]
pub fn payload_last_ordering(a: PayloadLast<Ordering>) -> isize {
    // CHECK-LABEL: @payload_last_ordering(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = sext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp sge i8 %a, 2
    // CHECK-NEXT: %[[ASSUME:.+]] = icmp ne i64 %0, 4
    // CHECK-NEXT: call void @llvm.assume(i1 %[[ASSUME]])
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 4
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], -2
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

// For a 1/2/3 payload, it would be nice if the zero niche were prioritized
// (particularly as that would let this test the 4th case), but layout actually
// puts the niches after the payload here too, so the niches are 4 and 5.
// - with the payload first, the payload variant equivalent is 3, which is a valid value.
// - with the payload last, the payload variant equivalent is 6, which we assume away.

pub enum OneTwoThree {
    One = 1,
    Two = 2,
    Three = 3,
}

#[unsafe(no_mangle)]
pub fn payload_first_123(a: PayloadFirst<OneTwoThree>) -> isize {
    // CHECK-LABEL: @payload_first_123(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = zext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp uge i8 %a, 4
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 3
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], -3
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

#[unsafe(no_mangle)]
pub fn payload_last_123(a: PayloadLast<OneTwoThree>) -> isize {
    // CHECK-LABEL: @payload_last_123(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = zext i8 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp uge i8 %a, 4
    // CHECK-NEXT: %[[ASSUME:.+]] = icmp ne i64 %0, 6
    // CHECK-NEXT: call void @llvm.assume(i1 %[[ASSUME]])
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 6
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], -4
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

// For a -4/-3/-2 payload, the niche start is negative so we need to be careful
// that it also gets sign-extended.  The niches are -1 and 0.
// - with the payload first, the payload variant equivalent is -2, which is a valid value.
// - with the payload last, the payload variant equivalent is 1, which we assume away.

#[repr(i16)]
pub enum Neg16Bit {
    NegFour = -4,
    NegThree = -3,
    NegTwo = -2,
}

#[unsafe(no_mangle)]
pub fn payload_first_neg16bit(a: PayloadFirst<Neg16Bit>) -> isize {
    // CHECK-LABEL: @payload_first_neg16bit(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = sext i16 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp sge i16 %a, -1
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 -2
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], 2
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}

#[unsafe(no_mangle)]
pub fn payload_last_neg16bit(a: PayloadLast<Neg16Bit>) -> isize {
    // CHECK-LABEL: @payload_last_neg16bit(
    // CHECK-NEXT: start:
    // CHECK-NEXT: %[[A_EXT:.+]] = sext i16 %a to i64
    // CHECK-NEXT: %[[IS_NICHE:.+]] = icmp sge i16 %a, -1
    // CHECK-NEXT: %[[ASSUME:.+]] = icmp ne i64 %0, 1
    // CHECK-NEXT: call void @llvm.assume(i1 %[[ASSUME]])
    // CHECK-NEXT: %[[ADJ_DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[A_EXT]], i64 1
    // CHECK-NEXT: %[[DISCR:.+]] = add i64 %[[ADJ_DISCR]], 1
    // CHECK-NEXT: ret i64 %[[DISCR]]

    discriminant_value(&a) as _
}
