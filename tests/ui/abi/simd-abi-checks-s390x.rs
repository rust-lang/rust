//@ add-core-stubs
//@ revisions: z10 z13_no_vector z13_soft_float
//@ build-fail
//@[z10] compile-flags: --target s390x-unknown-linux-gnu -C target-cpu=z10
//@[z10] needs-llvm-components: systemz
//@[z13_no_vector] compile-flags: --target s390x-unknown-linux-gnu -C target-cpu=z13 -C target-feature=-vector
//@[z13_no_vector] needs-llvm-components: systemz
// FIXME: +soft-float itself doesn't set -vector
//@[z13_soft_float] compile-flags: --target s390x-unknown-linux-gnu -C target-cpu=z13 -C target-feature=-vector,+soft-float
//@[z13_soft_float] needs-llvm-components: systemz

#![feature(no_core, repr_simd, s390x_target_feature)]
#![no_core]
#![crate_type = "lib"]
#![allow(non_camel_case_types, improper_ctypes_definitions)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct i8x8([i8; 8]);
#[repr(simd)]
pub struct i8x16([i8; 16]);
#[repr(simd)]
pub struct i8x32([i8; 32]);
#[repr(C)]
pub struct Wrapper<T>(T);
#[repr(transparent)]
pub struct TransparentWrapper<T>(T);

impl Copy for i8x8 {}
impl Copy for i8x16 {}
impl Copy for i8x32 {}
impl<T: Copy> Copy for Wrapper<T> {}
impl<T: Copy> Copy for TransparentWrapper<T> {}

#[no_mangle]
extern "C" fn vector_ret_small(x: &i8x8) -> i8x8 {
    //~^ ERROR requires the `vector` target feature, which is not enabled
    *x
}
#[no_mangle]
extern "C" fn vector_ret(x: &i8x16) -> i8x16 {
    //~^ ERROR requires the `vector` target feature, which is not enabled
    *x
}
#[no_mangle]
extern "C" fn vector_ret_large(x: &i8x32) -> i8x32 {
    // Ok
    *x
}

#[no_mangle]
#[target_feature(enable = "vector")]
unsafe extern "C" fn vector_ret_target_feature_small(x: &i8x8) -> i8x8 {
    // Ok
    *x
}
#[no_mangle]
#[target_feature(enable = "vector")]
unsafe extern "C" fn vector_target_feature_ret(x: &i8x16) -> i8x16 {
    // Ok
    *x
}
#[no_mangle]
#[target_feature(enable = "vector")]
unsafe extern "C" fn vector_ret_target_feature_large(x: &i8x32) -> i8x32 {
    // Ok
    *x
}

#[no_mangle]
extern "C" fn vector_wrapper_ret_small(x: &Wrapper<i8x8>) -> Wrapper<i8x8> {
    // Ok
    *x
}
#[no_mangle]
extern "C" fn vector_wrapper_ret(x: &Wrapper<i8x16>) -> Wrapper<i8x16> {
    // Ok
    *x
}
#[no_mangle]
extern "C" fn vector_wrapper_ret_large(x: &Wrapper<i8x32>) -> Wrapper<i8x32> {
    // Ok
    *x
}

#[no_mangle]
extern "C" fn vector_transparent_wrapper_ret_small(
    x: &TransparentWrapper<i8x8>,
) -> TransparentWrapper<i8x8> {
    //~^^^ ERROR requires the `vector` target feature, which is not enabled
    *x
}
#[no_mangle]
extern "C" fn vector_transparent_wrapper_ret(
    x: &TransparentWrapper<i8x16>,
) -> TransparentWrapper<i8x16> {
    //~^^^ ERROR requires the `vector` target feature, which is not enabled
    *x
}
#[no_mangle]
extern "C" fn vector_transparent_wrapper_ret_large(
    x: &TransparentWrapper<i8x32>,
) -> TransparentWrapper<i8x32> {
    // Ok
    *x
}

#[no_mangle]
extern "C" fn vector_arg_small(x: i8x8) -> i64 {
    //~^ ERROR requires the `vector` target feature, which is not enabled
    unsafe { *(&x as *const i8x8 as *const i64) }
}
#[no_mangle]
extern "C" fn vector_arg(x: i8x16) -> i64 {
    //~^ ERROR requires the `vector` target feature, which is not enabled
    unsafe { *(&x as *const i8x16 as *const i64) }
}
#[no_mangle]
extern "C" fn vector_arg_large(x: i8x32) -> i64 {
    // Ok
    unsafe { *(&x as *const i8x32 as *const i64) }
}

#[no_mangle]
extern "C" fn vector_wrapper_arg_small(x: Wrapper<i8x8>) -> i64 {
    //~^ ERROR requires the `vector` target feature, which is not enabled
    unsafe { *(&x as *const Wrapper<i8x8> as *const i64) }
}
#[no_mangle]
extern "C" fn vector_wrapper_arg(x: Wrapper<i8x16>) -> i64 {
    //~^ ERROR requires the `vector` target feature, which is not enabled
    unsafe { *(&x as *const Wrapper<i8x16> as *const i64) }
}
#[no_mangle]
extern "C" fn vector_wrapper_arg_large(x: Wrapper<i8x32>) -> i64 {
    // Ok
    unsafe { *(&x as *const Wrapper<i8x32> as *const i64) }
}

#[no_mangle]
extern "C" fn vector_transparent_wrapper_arg_small(x: TransparentWrapper<i8x8>) -> i64 {
    //~^ ERROR requires the `vector` target feature, which is not enabled
    unsafe { *(&x as *const TransparentWrapper<i8x8> as *const i64) }
}
#[no_mangle]
extern "C" fn vector_transparent_wrapper_arg(x: TransparentWrapper<i8x16>) -> i64 {
    //~^ ERROR requires the `vector` target feature, which is not enabled
    unsafe { *(&x as *const TransparentWrapper<i8x16> as *const i64) }
}
#[no_mangle]
extern "C" fn vector_transparent_wrapper_arg_large(x: TransparentWrapper<i8x32>) -> i64 {
    // Ok
    unsafe { *(&x as *const TransparentWrapper<i8x32> as *const i64) }
}
