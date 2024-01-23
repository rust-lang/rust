// compile-flags: --crate-type=lib -Zmerge-functions=disabled -O

#![feature(core_intrinsics)]

use std::intrinsics::is_val_statically_known;

pub struct A(u32);
pub enum B {
    Ye(u32),
}

#[inline]
pub fn _u32(a: u32) -> i32 {
    if unsafe { is_val_statically_known(a) } { 1 } else { 0 }
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
    if unsafe { is_val_statically_known(b) } { 3 } else { 2 }
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
