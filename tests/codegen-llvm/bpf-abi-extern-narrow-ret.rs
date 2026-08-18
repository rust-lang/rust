//@ add-minicore
//@ needs-llvm-components: bpf
//@ compile-flags: --target bpfel-unknown-none

#![feature(no_core)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

unsafe extern "C" {
    fn c_ret_u8() -> u8;
    fn c_ret_u16() -> u16;
    fn c_ret_i8() -> i8;
    fn c_ret_i16() -> i16;
}

// CHECK-LABEL: define {{.*}} @observe_u8(
// CHECK: call {{(noundef )?}}i8 @c_ret_u8()
#[unsafe(no_mangle)]
pub unsafe extern "C" fn observe_u8() -> u64 {
    c_ret_u8() as u64
}

// CHECK-LABEL: define {{.*}} @observe_u16(
// CHECK: call {{(noundef )?}}i16 @c_ret_u16()
#[unsafe(no_mangle)]
pub unsafe extern "C" fn observe_u16() -> u64 {
    c_ret_u16() as u64
}

// CHECK-LABEL: define {{.*}} @observe_i8(
// CHECK: call {{(noundef )?}}i8 @c_ret_i8()
#[unsafe(no_mangle)]
pub unsafe extern "C" fn observe_i8() -> i64 {
    c_ret_i8() as i64
}

// CHECK-LABEL: define {{.*}} @observe_i16(
// CHECK: call {{(noundef )?}}i16 @c_ret_i16()
#[unsafe(no_mangle)]
pub unsafe extern "C" fn observe_i16() -> i64 {
    c_ret_i16() as i64
}

// CHECK: declare {{(noundef )?}}i8 @c_ret_u8()
// CHECK: declare {{(noundef )?}}i16 @c_ret_u16()
// CHECK: declare {{(noundef )?}}i8 @c_ret_i8()
// CHECK: declare {{(noundef )?}}i16 @c_ret_i16()
