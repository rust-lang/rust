//! Make sure the compiler knows the result range of `ilog10`.

//@ compile-flags: -O -Z merge-functions=disabled

#![crate_type = "lib"]

use std::num::NonZero;

// Signed integers.

#[no_mangle]
fn i8_ilog10_range(value: i8) {
    const MAX_RESULT: u32 = i8::MAX.ilog10();

    // CHECK-LABEL: @i8_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value <= 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn i16_ilog10_range(value: i16) {
    const MAX_RESULT: u32 = i16::MAX.ilog10();

    // CHECK-LABEL: @i16_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value <= 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn i32_ilog10_range(value: i32) {
    const MAX_RESULT: u32 = i32::MAX.ilog10();

    // CHECK-LABEL: @i32_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value <= 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn i64_ilog10_range(value: i64) {
    const MAX_RESULT: u32 = i64::MAX.ilog10();

    // CHECK-LABEL: @i64_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value <= 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn i128_ilog10_range(value: i128) {
    const MAX_RESULT: u32 = i128::MAX.ilog10();

    // CHECK-LABEL: @i128_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value <= 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn isize_ilog10_range(value: isize) {
    const MAX_RESULT: u32 = isize::MAX.ilog10();

    // CHECK-LABEL: @isize_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value <= 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

// Unsigned integer types.

#[no_mangle]
fn u8_ilog10_range(value: u8) {
    const MAX_RESULT: u32 = u8::MAX.ilog10();

    // CHECK-LABEL: @u8_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value == 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn u16_ilog10_range(value: u16) {
    const MAX_RESULT: u32 = u16::MAX.ilog10();

    // CHECK-LABEL: @u16_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value == 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn u32_ilog10_range(value: u32) {
    const MAX_RESULT: u32 = u32::MAX.ilog10();

    // CHECK-LABEL: @u32_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value == 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn u64_ilog10_range(value: u64) {
    const MAX_RESULT: u32 = u64::MAX.ilog10();

    // CHECK-LABEL: @u64_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value == 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn u128_ilog10_range(value: u128) {
    const MAX_RESULT: u32 = u128::MAX.ilog10();

    // CHECK-LABEL: @u128_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value == 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

#[no_mangle]
fn usize_ilog10_range(value: usize) {
    const MAX_RESULT: u32 = usize::MAX.ilog10();

    // CHECK-LABEL: @usize_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value == 0 || value.ilog10() <= MAX_RESULT);
    assert!(value.checked_ilog10().is_none_or(|result| result <= MAX_RESULT));
}

// Signed non-zero integers do not have `ilog10` methods.

// Unsigned non-zero integers.

#[no_mangle]
fn non_zero_u8_ilog10_range(value: NonZero<u8>) {
    // CHECK-LABEL: @non_zero_u8_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value.ilog10() <= const { u8::MAX.ilog10() });
}

#[no_mangle]
fn non_zero_u16_ilog10_range(value: NonZero<u16>) {
    // CHECK-LABEL: @non_zero_u16_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value.ilog10() <= const { u16::MAX.ilog10() });
}

#[no_mangle]
fn non_zero_u32_ilog10_range(value: NonZero<u32>) {
    // CHECK-LABEL: @non_zero_u32_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value.ilog10() <= const { u32::MAX.ilog10() });
}

#[no_mangle]
fn non_zero_u64_ilog10_range(value: NonZero<u64>) {
    // CHECK-LABEL: @non_zero_u64_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value.ilog10() <= const { u64::MAX.ilog10() });
}

#[no_mangle]
fn non_zero_u128_ilog10_range(value: NonZero<u128>) {
    // CHECK-LABEL: @non_zero_u128_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value.ilog10() <= const { u128::MAX.ilog10() });
}

#[no_mangle]
fn non_zero_usize_ilog10_range(value: NonZero<usize>) {
    // CHECK-LABEL: @non_zero_usize_ilog10_range(
    // CHECK-NOT: panic
    // CHECK: ret void
    // CHECK-NEXT: }
    assert!(value.ilog10() <= const { usize::MAX.ilog10() });
}
