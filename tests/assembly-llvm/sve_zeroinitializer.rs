//@ assembly-output: emit-asm
//@ needs-llvm-components: aarch64
//@ compile-flags: --target aarch64-unknown-linux-gnu -C target-feature=+sve

#![allow(incomplete_features, internal_features)]
#![feature(simd_ffi, rustc_attrs, link_llvm_intrinsics, core_intrinsics)]
#![crate_type = "lib"]

#[rustc_scalable_vector(16)]
#[allow(non_camel_case_types)]
pub struct svbool_t(bool);

#[rustc_scalable_vector(8)]
#[allow(non_camel_case_types)]
pub struct svint16_t(i16);

#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
pub struct svfloat32_t(f32);

#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
pub struct svfloat32x2_t(svfloat32_t, svfloat32_t);

#[target_feature(enable = "sve")]
fn svbool_zeroinitializer() -> svbool_t {
    unsafe { std::intrinsics::simd::scalable::sve_zeroinitializer() }
}

#[target_feature(enable = "sve")]
pub fn svint16_zeroinitializer() -> svint16_t {
    unsafe { std::intrinsics::simd::scalable::sve_zeroinitializer() }
}

#[target_feature(enable = "sve")]
pub fn svfloat32x2_zeroinitializer() -> svfloat32x2_t {
    unsafe { std::intrinsics::simd::scalable::sve_zeroinitializer() }
}

// CHECK-LABEL: svbool_false
#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe extern "C" fn svbool_false() -> svbool_t {
    // CHECK: pfalse p0.b
    svbool_zeroinitializer()
}

// CHECK-LABEL: svint_zero
#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe extern "C" fn svint_zero() -> svint16_t {
    // CHECK: movi v0.2d, #0000000000000000
    svint16_zeroinitializer()
}

// CHECK-LABEL: svfloat_tuple_zero
#[no_mangle]
#[target_feature(enable = "sve")]
pub unsafe extern "C" fn svfloat_tuple_zero() -> svfloat32x2_t {
    // CHECK: movi v0.2d, #0000000000000000
    svfloat32x2_zeroinitializer()
}
