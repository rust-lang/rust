//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
// 32-bit x86 returns `f32` and `f64` differently to avoid the x87 stack.
//@ revisions: x86 other
//@[x86] only-rustc_abi-x86-sse2
//@[other] ignore-x86

#![crate_type = "lib"]

#[no_mangle]
pub struct F32(f32);

// other: define{{.*}}float @add_newtype_f32(float %a, float %b)
// x86: define{{.*}}<4 x i8> @add_newtype_f32(float %a, float %b)
#[inline(never)]
#[no_mangle]
pub fn add_newtype_f32(a: F32, b: F32) -> F32 {
    F32(a.0 + b.0)
}

#[no_mangle]
pub struct F64(f64);

// other: define{{.*}}double @add_newtype_f64(double %a, double %b)
// x86: define{{.*}}<8 x i8> @add_newtype_f64(double %a, double %b)
#[inline(never)]
#[no_mangle]
pub fn add_newtype_f64(a: F64, b: F64) -> F64 {
    F64(a.0 + b.0)
}
