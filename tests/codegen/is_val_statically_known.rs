//@ compile-flags: --crate-type=lib -Zmerge-functions=disabled -Copt-level=3

#![feature(core_intrinsics)]
#![feature(f16, f128)]

use std::intrinsics::is_val_statically_known;

pub struct A(u32);
pub enum B {
    Ye(u32),
}

#[inline]
pub fn _u32(a: u32) -> i32 {
    if is_val_statically_known(a) { 1 } else { 0 }
}

// CHECK-LABEL: @_u32_true(
#[no_mangle]
pub fn _u32_true() -> i32 {
    // CHECK: ret i32 1
    _u32(1)
}

// CHECK-LABEL: @_u32_false(
#[no_mangle]
pub fn _u32_false(a: u32) -> i32 {
    // CHECK: ret i32 0
    _u32(a)
}

#[inline]
pub fn _bool(b: bool) -> i32 {
    if is_val_statically_known(b) { 3 } else { 2 }
}

// CHECK-LABEL: @_bool_true(
#[no_mangle]
pub fn _bool_true() -> i32 {
    // CHECK: ret i32 3
    _bool(true)
}

// CHECK-LABEL: @_bool_false(
#[no_mangle]
pub fn _bool_false(b: bool) -> i32 {
    // CHECK: ret i32 2
    _bool(b)
}

#[inline]
pub fn _iref(a: &u8) -> i32 {
    if is_val_statically_known(a) { 5 } else { 4 }
}

// CHECK-LABEL: @_iref_borrow(
#[no_mangle]
pub fn _iref_borrow() -> i32 {
    // CHECK: ret i32 4
    _iref(&0)
}

// CHECK-LABEL: @_iref_arg(
#[no_mangle]
pub fn _iref_arg(a: &u8) -> i32 {
    // CHECK: ret i32 4
    _iref(a)
}

#[inline]
pub fn _slice_ref(a: &[u8]) -> i32 {
    if is_val_statically_known(a) { 7 } else { 6 }
}

// CHECK-LABEL: @_slice_ref_borrow(
#[no_mangle]
pub fn _slice_ref_borrow() -> i32 {
    // CHECK: ret i32 6
    _slice_ref(&[0; 3])
}

// CHECK-LABEL: @_slice_ref_arg(
#[no_mangle]
pub fn _slice_ref_arg(a: &[u8]) -> i32 {
    // CHECK: ret i32 6
    _slice_ref(a)
}

#[inline]
pub fn _f16(a: f16) -> i32 {
    if is_val_statically_known(a) { 1 } else { 0 }
}

// CHECK-LABEL: @_f16_true(
#[no_mangle]
pub fn _f16_true() -> i32 {
    // CHECK: ret i32 1
    _f16(1.0)
}

// CHECK-LABEL: @_f16_false(
#[no_mangle]
pub fn _f16_false(a: f16) -> i32 {
    // CHECK: ret i32 0
    _f16(a)
}

#[inline]
pub fn _f32(a: f32) -> i32 {
    if is_val_statically_known(a) { 1 } else { 0 }
}

// CHECK-LABEL: @_f32_true(
#[no_mangle]
pub fn _f32_true() -> i32 {
    // CHECK: ret i32 1
    _f32(1.0)
}

// CHECK-LABEL: @_f32_false(
#[no_mangle]
pub fn _f32_false(a: f32) -> i32 {
    // CHECK: ret i32 0
    _f32(a)
}

#[inline]
pub fn _f64(a: f64) -> i32 {
    if is_val_statically_known(a) { 1 } else { 0 }
}

// CHECK-LABEL: @_f64_true(
#[no_mangle]
pub fn _f64_true() -> i32 {
    // CHECK: ret i32 1
    _f64(1.0)
}

// CHECK-LABEL: @_f64_false(
#[no_mangle]
pub fn _f64_false(a: f64) -> i32 {
    // CHECK: ret i32 0
    _f64(a)
}

#[inline]
pub fn _f128(a: f128) -> i32 {
    if is_val_statically_known(a) { 1 } else { 0 }
}

// CHECK-LABEL: @_f128_true(
#[no_mangle]
pub fn _f128_true() -> i32 {
    // CHECK: ret i32 1
    _f128(1.0)
}

// CHECK-LABEL: @_f128_false(
#[no_mangle]
pub fn _f128_false(a: f128) -> i32 {
    // CHECK: ret i32 0
    _f128(a)
}
