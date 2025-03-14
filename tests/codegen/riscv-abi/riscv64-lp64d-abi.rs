//@ add-core-stubs
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes --target riscv64gc-unknown-linux-gnu
//@ needs-llvm-components: riscv

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: define void @f_fpr_tracking(double %0, double %1, double %2, double %3, double %4, double %5, double %6, double %7, i8 noundef zeroext %i)
#[no_mangle]
pub extern "C" fn f_fpr_tracking(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    i: u8,
) {
}

#[repr(C)]
pub struct Double {
    f: f64,
}

#[repr(C)]
pub struct DoubleDouble {
    f: f64,
    g: f64,
}

#[repr(C)]
pub struct DoubleFloat {
    f: f64,
    g: f32,
}

// CHECK: define void @f_double_s_arg(double %0)
#[no_mangle]
pub extern "C" fn f_double_s_arg(a: Double) {}

// CHECK: define double @f_ret_double_s()
#[no_mangle]
pub extern "C" fn f_ret_double_s() -> Double {
    Double { f: 1. }
}

// CHECK: define void @f_double_double_s_arg({ double, double } %0)
#[no_mangle]
pub extern "C" fn f_double_double_s_arg(a: DoubleDouble) {}

// CHECK: define { double, double } @f_ret_double_double_s()
#[no_mangle]
pub extern "C" fn f_ret_double_double_s() -> DoubleDouble {
    DoubleDouble { f: 1., g: 2. }
}

// CHECK: define void @f_double_float_s_arg({ double, float } %0)
#[no_mangle]
pub extern "C" fn f_double_float_s_arg(a: DoubleFloat) {}

// CHECK: define { double, float } @f_ret_double_float_s()
#[no_mangle]
pub extern "C" fn f_ret_double_float_s() -> DoubleFloat {
    DoubleFloat { f: 1., g: 2. }
}

// CHECK: define void @f_double_double_s_arg_insufficient_fprs(double %0, double %1, double %2, double %3, double %4, double %5, double %6, [2 x i64] %7)
#[no_mangle]
pub extern "C" fn f_double_double_s_arg_insufficient_fprs(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: DoubleDouble,
) {
}

#[repr(C)]
pub struct DoubleInt8 {
    f: f64,
    i: i8,
}

#[repr(C)]
pub struct DoubleUInt8 {
    f: f64,
    i: u8,
}

#[repr(C)]
pub struct DoubleInt32 {
    f: f64,
    i: i32,
}

#[repr(C)]
pub struct DoubleInt64 {
    f: f64,
    i: i64,
}

// CHECK: define void @f_double_int8_s_arg({ double, i8 } %0)
#[no_mangle]
pub extern "C" fn f_double_int8_s_arg(a: DoubleInt8) {}

// CHECK: define { double, i8 } @f_ret_double_int8_s()
#[no_mangle]
pub extern "C" fn f_ret_double_int8_s() -> DoubleInt8 {
    DoubleInt8 { f: 1., i: 2 }
}

// CHECK: define void @f_double_int32_s_arg({ double, i32 } %0)
#[no_mangle]
pub extern "C" fn f_double_int32_s_arg(a: DoubleInt32) {}

// CHECK: define { double, i32 } @f_ret_double_int32_s()
#[no_mangle]
pub extern "C" fn f_ret_double_int32_s() -> DoubleInt32 {
    DoubleInt32 { f: 1., i: 2 }
}

// CHECK: define void @f_double_uint8_s_arg({ double, i8 } %0)
#[no_mangle]
pub extern "C" fn f_double_uint8_s_arg(a: DoubleUInt8) {}

// CHECK: define { double, i8 } @f_ret_double_uint8_s()
#[no_mangle]
pub extern "C" fn f_ret_double_uint8_s() -> DoubleUInt8 {
    DoubleUInt8 { f: 1., i: 2 }
}

// CHECK: define void @f_double_int64_s_arg({ double, i64 } %0)
#[no_mangle]
pub extern "C" fn f_double_int64_s_arg(a: DoubleInt64) {}

// CHECK: define { double, i64 } @f_ret_double_int64_s()
#[no_mangle]
pub extern "C" fn f_ret_double_int64_s() -> DoubleInt64 {
    DoubleInt64 { f: 1., i: 2 }
}

// CHECK: define void @f_double_int8_s_arg_insufficient_gprs(i32 noundef signext %a, i32 noundef signext %b, i32 noundef signext %c, i32 noundef signext %d, i32 noundef signext %e, i32 noundef signext %f, i32 noundef signext %g, i32 noundef signext %h, [2 x i64] %0)
#[no_mangle]
pub extern "C" fn f_double_int8_s_arg_insufficient_gprs(
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    e: i32,
    f: i32,
    g: i32,
    h: i32,
    i: DoubleInt8,
) {
}

// CHECK: define void @f_struct_double_int8_insufficient_fprs(float %0, double %1, double %2, double %3, double %4, double %5, double %6, double %7, [2 x i64] %8)
#[no_mangle]
pub extern "C" fn f_struct_double_int8_insufficient_fprs(
    a: f32,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    i: DoubleInt8,
) {
}

#[repr(C)]
pub struct DoubleArr1 {
    a: [f64; 1],
}

// CHECK: define void @f_doublearr1_s_arg(double %0)
#[no_mangle]
pub extern "C" fn f_doublearr1_s_arg(a: DoubleArr1) {}

// CHECK: define double @f_ret_doublearr1_s()
#[no_mangle]
pub extern "C" fn f_ret_doublearr1_s() -> DoubleArr1 {
    DoubleArr1 { a: [1.] }
}

#[repr(C)]
pub struct DoubleArr2 {
    a: [f64; 2],
}

// CHECK: define void @f_doublearr2_s_arg({ double, double } %0)
#[no_mangle]
pub extern "C" fn f_doublearr2_s_arg(a: DoubleArr2) {}

// CHECK: define { double, double } @f_ret_doublearr2_s()
#[no_mangle]
pub extern "C" fn f_ret_doublearr2_s() -> DoubleArr2 {
    DoubleArr2 { a: [1., 2.] }
}

#[repr(C)]
pub struct Tricky1 {
    f: [f64; 1],
}

#[repr(C)]
pub struct DoubleArr2Tricky1 {
    g: [Tricky1; 2],
}

// CHECK: define void @f_doublearr2_tricky1_s_arg({ double, double } %0)
#[no_mangle]
pub extern "C" fn f_doublearr2_tricky1_s_arg(a: DoubleArr2Tricky1) {}

// CHECK: define { double, double } @f_ret_doublearr2_tricky1_s()
#[no_mangle]
pub extern "C" fn f_ret_doublearr2_tricky1_s() -> DoubleArr2Tricky1 {
    DoubleArr2Tricky1 { g: [Tricky1 { f: [1.] }, Tricky1 { f: [2.] }] }
}

#[repr(C)]
pub struct EmptyStruct {}

#[repr(C)]
pub struct DoubleArr2Tricky2 {
    s: EmptyStruct,
    g: [Tricky1; 2],
}

// CHECK: define void @f_doublearr2_tricky2_s_arg({ double, double } %0)
#[no_mangle]
pub extern "C" fn f_doublearr2_tricky2_s_arg(a: DoubleArr2Tricky2) {}

// CHECK: define { double, double } @f_ret_doublearr2_tricky2_s()
#[no_mangle]
pub extern "C" fn f_ret_doublearr2_tricky2_s() -> DoubleArr2Tricky2 {
    DoubleArr2Tricky2 { s: EmptyStruct {}, g: [Tricky1 { f: [1.] }, Tricky1 { f: [2.] }] }
}

#[repr(C)]
pub struct IntDoubleInt {
    a: i32,
    b: f64,
    c: i32,
}

// CHECK: define void @f_int_double_int_s_arg(ptr {{.*}} %a)
#[no_mangle]
pub extern "C" fn f_int_double_int_s_arg(a: IntDoubleInt) {}

// CHECK: define void @f_ret_int_double_int_s(ptr {{.*}} sret([24 x i8]) align 8 {{.*}}dereferenceable(24) %_0)
#[no_mangle]
pub extern "C" fn f_ret_int_double_int_s() -> IntDoubleInt {
    IntDoubleInt { a: 1, b: 2., c: 3 }
}

#[repr(C)]
pub struct CharCharDouble {
    a: u8,
    b: u8,
    c: f64,
}

// CHECK: define void @f_char_char_double_s_arg([2 x i64] %0)
#[no_mangle]
pub extern "C" fn f_char_char_double_s_arg(a: CharCharDouble) {}

// CHECK: define [2 x i64] @f_ret_char_char_double_s()
#[no_mangle]
pub extern "C" fn f_ret_char_char_double_s() -> CharCharDouble {
    CharCharDouble { a: 1, b: 2, c: 3. }
}

#[repr(C)]
pub union DoubleU {
    a: f64,
}

// CHECK: define void @f_double_u_arg(i64 %0)
#[no_mangle]
pub extern "C" fn f_double_u_arg(a: DoubleU) {}

// CHECK: define i64 @f_ret_double_u()
#[no_mangle]
pub extern "C" fn f_ret_double_u() -> DoubleU {
    unsafe { DoubleU { a: 1. } }
}
