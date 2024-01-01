// compile-flags: -C opt-level=1 -Z merge-functions=disabled

#![crate_type = "lib"]

use core::num::{
    NonZeroI128, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI8, NonZeroIsize, NonZeroU128,
    NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize,
};

extern crate core;

// CHECK-LABEL: void @non_zero_i8_is_not_zero(i8
#[no_mangle]
fn non_zero_i8_is_not_zero(x: NonZeroI8) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(i8::from(x), 0);
}

// CHECK-LABEL: void @non_zero_i16_is_not_zero(i16
#[no_mangle]
fn non_zero_i16_is_not_zero(x: NonZeroI16) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(i16::from(x), 0);
}

// CHECK-LABEL: void @non_zero_i32_is_not_zero(i32
#[no_mangle]
fn non_zero_i32_is_not_zero(x: NonZeroI32) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(i32::from(x), 0);
}

// CHECK-LABEL: void @non_zero_i64_is_not_zero(i64
#[no_mangle]
fn non_zero_i64_is_not_zero(x: NonZeroI64) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(i64::from(x), 0);
}

// CHECK-LABEL: void @non_zero_i128_is_not_zero(i128
#[no_mangle]
fn non_zero_i128_is_not_zero(x: NonZeroI128) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(i128::from(x), 0);
}

// CHECK-LABEL: void @non_zero_isize_is_not_zero(i
#[no_mangle]
fn non_zero_isize_is_not_zero(x: NonZeroIsize) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(isize::from(x), 0);
}

// CHECK-LABEL: void @non_zero_u8_is_not_zero(i8
#[no_mangle]
fn non_zero_u8_is_not_zero(x: NonZeroU8) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(u8::from(x), 0);
}

// CHECK-LABEL: void @non_zero_u16_is_not_zero(i16
#[no_mangle]
fn non_zero_u16_is_not_zero(x: NonZeroU16) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(u16::from(x), 0);
}

// CHECK-LABEL: void @non_zero_u32_is_not_zero(i32
#[no_mangle]
fn non_zero_u32_is_not_zero(x: NonZeroU32) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(u32::from(x), 0);
}

// CHECK-LABEL: void @non_zero_u64_is_not_zero(i64
#[no_mangle]
fn non_zero_u64_is_not_zero(x: NonZeroU64) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(u64::from(x), 0);
}

// CHECK-LABEL: void @non_zero_u128_is_not_zero(i128
#[no_mangle]
fn non_zero_u128_is_not_zero(x: NonZeroU128) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(u128::from(x), 0);
}

// CHECK-LABEL: void @non_zero_usize_is_not_zero(i
#[no_mangle]
fn non_zero_usize_is_not_zero(x: NonZeroUsize) {
    // CHECK-NOT: br
    assert_ne!(x.get(), 0);
    assert_ne!(usize::from(x), 0);
}
