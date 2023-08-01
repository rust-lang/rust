// compile-flags: --crate-type=lib
#![feature(core_intrinsics)]

use std::intrinsics::is_constant;

pub struct A(u32);
pub enum B {
    Ye(u32),
}

#[inline]
pub fn tuple_struct(a: A) -> i32 {
    if is_constant(a) { 1 } else { 0 }
}

// CHECK-LABEL: @tuple_struct_true(
#[no_mangle]
pub fn tuple_struct_true() -> i32 {
    // CHECK: ret i32 1
    tuple_struct(A(1))
}

// CHECK-LABEL: @tuple_struct_false(
#[no_mangle]
pub fn tuple_struct_false(a: A) -> i32 {
    // CHECK: ret i32 0
    tuple_struct(a)
}

#[inline]
pub fn r#enum(b: B) -> i32 {
    if is_constant(b) { 3 } else { 2 }
}

// CHECK-LABEL: @enum_true(
#[no_mangle]
pub fn enum_true() -> i32 {
    // CHECK: ret i32 3
    r#enum(B::Ye(2))
}

// CHECK-LABEL: @enum_false(
#[no_mangle]
pub fn enum_false(b: B) -> i32 {
    // CHECK: ret i32 2
    r#enum(b)
}
