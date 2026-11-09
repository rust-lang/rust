//@ add-minicore
//@ compile-flags: --target i686-pc-windows-msvc -Copt-level=3
//@ needs-llvm-components: x86
#![crate_type = "lib"]
#![feature(no_core, abi_vectorcall)]
#![no_core]

// Test how vectors are passed by the win32 calling conventions.
//
// MSVC 2015+ passes the first 3 vector arguments in XMM registers. `extern "vectorcall"` has
// 6 XMM registers available, and also uses them for `f32` and `f64`.
//
// Vectors that don't get an XMM register are passed indirectly, and the pointer may be passed
// via a GPR (indicated by `inreg`) if one is available.

extern crate minicore;
use minicore::simd::{f32x4, f32x8};
use minicore::*;

// CHECK-LABEL: @cdecl_vec4(
// CHECK-SAME: <4 x float> {{.*}}%a, <4 x float> {{.*}}%b, <4 x float> {{.*}}%c, ptr {{.*}}%d, i32 {{.*}}%e)
#[unsafe(no_mangle)]
pub extern "C" fn cdecl_vec4(a: f32x4, b: f32x4, c: f32x4, d: f32x4, e: i32) {}

// CHECK-LABEL: @cdecl_f32x8(
// CHECK-SAME: <8 x float> {{.*}}%a, <8 x float> {{.*}}%b, <8 x float> {{.*}}%c, ptr {{.*}}%d)
#[unsafe(no_mangle)]
#[target_feature(enable = "avx")]
pub extern "C" fn cdecl_f32x8(a: f32x8, b: f32x8, c: f32x8, d: f32x8) {}

#[repr(C)]
pub struct Pair {
    a: f32x4,
    b: f32x4,
}

// CHECK-LABEL: @cdecl_aggregate(
// CHECK-SAME: ptr {{.*}}byval([32 x i8]) {{.*}}, <4 x float> {{.*}}%a, <4 x float> {{.*}}%b, <4 x float> {{.*}}%c, ptr {{.*}}%d)
#[unsafe(no_mangle)]
pub extern "C" fn cdecl_aggregate(_: Pair, a: f32x4, b: f32x4, c: f32x4, d: f32x4) {}

// CHECK-LABEL: @"\01@fastcall_vec4@68"(
// CHECK-SAME: <4 x float> {{.*}}%a, <4 x float> {{.*}}%b, <4 x float> {{.*}}%c, ptr {{.*}}inreg{{.*}}%d, i32 inreg {{.*}}%e)
#[unsafe(no_mangle)]
pub extern "fastcall" fn fastcall_vec4(a: f32x4, b: f32x4, c: f32x4, d: f32x4, e: i32) {}

// CHECK-LABEL: @"\01@fastcall_vec6@100"(
// CHECK-SAME: <4 x float> {{.*}}%a, <4 x float> {{.*}}%b, <4 x float> {{.*}}%c, ptr {{.*}}inreg{{.*}}%d, ptr {{.*}}inreg{{.*}}%e, ptr {{.*}}%f, i32 {{.*}}%g)
#[unsafe(no_mangle)]
pub extern "fastcall" fn fastcall_vec6(
    a: f32x4,
    b: f32x4,
    c: f32x4,
    d: f32x4,
    e: f32x4,
    f: f32x4,
    g: i32,
) {
}

// CHECK-LABEL: @"\01vectorcall_vec7@@116"(
// CHECK-SAME: <4 x float> {{.*}}%a, <4 x float> {{.*}}%b, <4 x float> {{.*}}%c, <4 x float> {{.*}}%d, <4 x float> {{.*}}%e, <4 x float> {{.*}}%f, ptr {{.*}}inreg{{.*}}%g, i32 inreg {{.*}}%h)
#[unsafe(no_mangle)]
pub extern "vectorcall" fn vectorcall_vec7(
    a: f32x4,
    b: f32x4,
    c: f32x4,
    d: f32x4,
    e: f32x4,
    f: f32x4,
    g: f32x4,
    h: i32,
) {
}

// Floats use vector registers too, but are passed on the stack when none are left.
//
// CHECK-LABEL: @"\01vectorcall_floats@@56"(
// CHECK-SAME: float {{.*}}%a, double {{.*}}%b, float {{.*}}%c, float {{.*}}%d, float {{.*}}%e, float {{.*}}%f, float {{.*}}%g, ptr {{.*}}inreg{{.*}}%h, float {{.*}}%i, i32 inreg {{.*}}%j)
#[unsafe(no_mangle)]
pub extern "vectorcall" fn vectorcall_floats(
    a: f32,
    b: f64,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32x4,
    i: f32,
    j: i32,
) {
}

// CHECK-LABEL: @"\01vectorcall_mixed@@92"(
// CHECK-SAME: float {{.*}}%a, <4 x float> {{.*}}%b, float {{.*}}%c, <4 x float> {{.*}}%d, float {{.*}}%e, <4 x float> {{.*}}%f, ptr {{.*}}inreg{{.*}}%g, ptr {{.*}}inreg{{.*}}%h)
#[unsafe(no_mangle)]
pub extern "vectorcall" fn vectorcall_mixed(
    a: f32,
    b: f32x4,
    c: f32,
    d: f32x4,
    e: f32,
    f: f32x4,
    g: f32x4,
    h: f32x4,
) {
}
