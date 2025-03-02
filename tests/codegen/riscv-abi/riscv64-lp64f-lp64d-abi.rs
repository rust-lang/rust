//@ add-core-stubs
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes --target riscv64gc-unknown-linux-gnu
//@ needs-llvm-components: riscv

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: define void @f_fpr_tracking(float %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, i8 noundef zeroext %i)
#[no_mangle]
pub extern "C" fn f_fpr_tracking(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    i: u8,
) {
}

#[repr(C)]
pub struct Float {
    f: f32,
}

#[repr(C)]
pub struct FloatFloat {
    f: f32,
    g: f32,
}

// CHECK: define void @f_float_s_arg(float %0)
#[no_mangle]
pub extern "C" fn f_float_s_arg(a: Float) {}

// CHECK: define float @f_ret_float_s()
#[no_mangle]
pub extern "C" fn f_ret_float_s() -> Float {
    Float { f: 1. }
}

// CHECK: define void @f_float_float_s_arg({ float, float } %0)
#[no_mangle]
pub extern "C" fn f_float_float_s_arg(a: FloatFloat) {}

// CHECK: define { float, float } @f_ret_float_float_s()
#[no_mangle]
pub extern "C" fn f_ret_float_float_s() -> FloatFloat {
    FloatFloat { f: 1., g: 2. }
}

// CHECK: define void @f_float_float_s_arg_insufficient_fprs(float %0, float %1, float %2, float %3, float %4, float %5, float %6, i64 %7)
#[no_mangle]
pub extern "C" fn f_float_float_s_arg_insufficient_fprs(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: FloatFloat,
) {
}

#[repr(C)]
pub struct FloatInt8 {
    f: f32,
    i: i8,
}

#[repr(C)]
pub struct FloatUInt8 {
    f: f32,
    i: u8,
}

#[repr(C)]
pub struct FloatInt32 {
    f: f32,
    i: i32,
}

#[repr(C)]
pub struct FloatInt64 {
    f: f32,
    i: i64,
}

// CHECK: define void @f_float_int8_s_arg({ float, i8 } %0)
#[no_mangle]
pub extern "C" fn f_float_int8_s_arg(a: FloatInt8) {}

// CHECK: define { float, i8 } @f_ret_float_int8_s()
#[no_mangle]
pub extern "C" fn f_ret_float_int8_s() -> FloatInt8 {
    FloatInt8 { f: 1., i: 2 }
}

// CHECK: define void @f_float_int32_s_arg({ float, i32 } %0)
#[no_mangle]
pub extern "C" fn f_float_int32_s_arg(a: FloatInt32) {}

// CHECK: define { float, i32 } @f_ret_float_int32_s()
#[no_mangle]
pub extern "C" fn f_ret_float_int32_s() -> FloatInt32 {
    FloatInt32 { f: 1., i: 2 }
}

// CHECK: define void @f_float_uint8_s_arg({ float, i8 } %0)
#[no_mangle]
pub extern "C" fn f_float_uint8_s_arg(a: FloatUInt8) {}

// CHECK: define { float, i8 } @f_ret_float_uint8_s()
#[no_mangle]
pub extern "C" fn f_ret_float_uint8_s() -> FloatUInt8 {
    FloatUInt8 { f: 1., i: 2 }
}

// CHECK: define void @f_float_int64_s_arg({ float, i64 } %0)
#[no_mangle]
pub extern "C" fn f_float_int64_s_arg(a: FloatInt64) {}

// CHECK: define { float, i64 } @f_ret_float_int64_s()
#[no_mangle]
pub extern "C" fn f_ret_float_int64_s() -> FloatInt64 {
    FloatInt64 { f: 1., i: 2 }
}

// CHECK: define void @f_float_int8_s_arg_insufficient_gprs(i32 noundef signext %a, i32 noundef signext %b, i32 noundef signext %c, i32 noundef signext %d, i32 noundef signext %e, i32 noundef signext %f, i32 noundef signext %g, i32 noundef signext %h, i64 %0)
#[no_mangle]
pub extern "C" fn f_float_int8_s_arg_insufficient_gprs(
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    e: i32,
    f: i32,
    g: i32,
    h: i32,
    i: FloatInt8,
) {
}

// CHECK: define void @f_struct_float_int8_insufficient_fprs(float %0, float %1, float %2,  float %3, float %4, float %5, float %6, float %7, i64 %8)
#[no_mangle]
pub extern "C" fn f_struct_float_int8_insufficient_fprs(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    i: FloatInt8,
) {
}

#[repr(C)]
pub struct FloatArr1 {
    a: [f32; 1],
}

// CHECK: define void @f_floatarr1_s_arg(float %0)
#[no_mangle]
pub extern "C" fn f_floatarr1_s_arg(a: FloatArr1) {}

// CHECK: define float @f_ret_floatarr1_s()
#[no_mangle]
pub extern "C" fn f_ret_floatarr1_s() -> FloatArr1 {
    FloatArr1 { a: [1.] }
}

#[repr(C)]
pub struct FloatArr2 {
    a: [f32; 2],
}

// CHECK: define void @f_floatarr2_s_arg({ float, float } %0)
#[no_mangle]
pub extern "C" fn f_floatarr2_s_arg(a: FloatArr2) {}

// CHECK: define { float, float } @f_ret_floatarr2_s()
#[no_mangle]
pub extern "C" fn f_ret_floatarr2_s() -> FloatArr2 {
    FloatArr2 { a: [1., 2.] }
}

#[repr(C)]
pub struct Tricky1 {
    f: [f32; 1],
}

#[repr(C)]
pub struct FloatArr2Tricky1 {
    g: [Tricky1; 2],
}

// CHECK: define void @f_floatarr2_tricky1_s_arg({ float, float } %0)
#[no_mangle]
pub extern "C" fn f_floatarr2_tricky1_s_arg(a: FloatArr2Tricky1) {}

// CHECK: define { float, float } @f_ret_floatarr2_tricky1_s()
#[no_mangle]
pub extern "C" fn f_ret_floatarr2_tricky1_s() -> FloatArr2Tricky1 {
    FloatArr2Tricky1 { g: [Tricky1 { f: [1.] }, Tricky1 { f: [2.] }] }
}

#[repr(C)]
pub struct EmptyStruct {}

#[repr(C)]
pub struct FloatArr2Tricky2 {
    s: EmptyStruct,
    g: [Tricky1; 2],
}

// CHECK: define void @f_floatarr2_tricky2_s_arg({ float, float } %0)
#[no_mangle]
pub extern "C" fn f_floatarr2_tricky2_s_arg(a: FloatArr2Tricky2) {}

// CHECK: define { float, float } @f_ret_floatarr2_tricky2_s()
#[no_mangle]
pub extern "C" fn f_ret_floatarr2_tricky2_s() -> FloatArr2Tricky2 {
    FloatArr2Tricky2 { s: EmptyStruct {}, g: [Tricky1 { f: [1.] }, Tricky1 { f: [2.] }] }
}

#[repr(C)]
pub struct IntFloatInt {
    a: i32,
    b: f32,
    c: i32,
}

// CHECK: define void @f_int_float_int_s_arg([2 x i64] %0)
#[no_mangle]
pub extern "C" fn f_int_float_int_s_arg(a: IntFloatInt) {}

// CHECK: define [2 x i64] @f_ret_int_float_int_s()
#[no_mangle]
pub extern "C" fn f_ret_int_float_int_s() -> IntFloatInt {
    IntFloatInt { a: 1, b: 2., c: 3 }
}

#[repr(C)]
pub struct CharCharFloat {
    a: u8,
    b: u8,
    c: f32,
}

// CHECK: define void @f_char_char_float_s_arg(i64 %0)
#[no_mangle]
pub extern "C" fn f_char_char_float_s_arg(a: CharCharFloat) {}

// CHECK: define i64 @f_ret_char_char_float_s()
#[no_mangle]
pub extern "C" fn f_ret_char_char_float_s() -> CharCharFloat {
    CharCharFloat { a: 1, b: 2, c: 3. }
}

#[repr(C)]
pub union FloatU {
    a: f32,
}

// CHECK: define void @f_float_u_arg(i64 %0)
#[no_mangle]
pub extern "C" fn f_float_u_arg(a: FloatU) {}

// CHECK: define i64 @f_ret_float_u()
#[no_mangle]
pub extern "C" fn f_ret_float_u() -> FloatU {
    unsafe { FloatU { a: 1. } }
}
