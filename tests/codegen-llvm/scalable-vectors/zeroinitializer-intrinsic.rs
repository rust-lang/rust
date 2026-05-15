//@ edition: 2021
//@ only-aarch64
#![crate_type = "lib"]
#![allow(incomplete_features, internal_features)]
#![feature(simd_ffi, rustc_attrs, link_llvm_intrinsics, core_intrinsics)]

#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
pub struct svbool_t(bool);

#[rustc_scalable_vector(8)]
#[allow(non_camel_case_types)]
pub struct svint16_t(i16);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
pub struct svfloat32_t(f32);

#[rustc_scalable_vector(2)]
#[allow(non_camel_case_types)]
pub struct svfloat64_t(f64);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
pub struct svfloat32x2_t(svfloat32_t, svfloat32_t);

#[no_mangle]
#[target_feature(enable = "sve")]
// CHECK-LABEL: svbool_zeroinitializer
pub fn svbool_zeroinitializer() -> svbool_t {
    // CHECK: start:
    // CHECK: ret <vscale x 16 x i1> zeroinitializer
    unsafe { std::intrinsics::simd::scalable::sve_zeroinitializer() }
}

#[no_mangle]
#[target_feature(enable = "sve")]
// CHECK-LABEL: svint16_zeroinitializer
pub fn svint16_zeroinitializer() -> svint16_t {
    // CHECK: start:
    // CHECK: ret <vscale x 8 x i16> zeroinitializer
    unsafe { std::intrinsics::simd::scalable::sve_zeroinitializer() }
}

#[no_mangle]
#[target_feature(enable = "sve")]
// CHECK-LABEL: svfloat32_zeroinitializer
pub fn svfloat32_zeroinitializer() -> svfloat32_t {
    // CHECK: start:
    // CHECK: ret <vscale x 4 x float> zeroinitializer
    unsafe { std::intrinsics::simd::scalable::sve_zeroinitializer() }
}

#[no_mangle]
#[target_feature(enable = "sve")]
// CHECK-LABEL: svfloat64_zeroinitializer
pub fn svfloat64_zeroinitializer() -> svfloat64_t {
    // CHECK: start:
    // CHECK: ret <vscale x 2 x double> zeroinitializer
    unsafe { std::intrinsics::simd::scalable::sve_zeroinitializer() }
}

#[no_mangle]
#[target_feature(enable = "sve")]
// CHECK-LABEL: svfloat32x2_zeroinitializer
pub fn svfloat32x2_zeroinitializer() -> svfloat32x2_t {
    // CHECK: start:
    // CHECK: ret { <vscale x 4 x float>, <vscale x 4 x float> } zeroinitializer
    unsafe { std::intrinsics::simd::scalable::sve_zeroinitializer() }
}
