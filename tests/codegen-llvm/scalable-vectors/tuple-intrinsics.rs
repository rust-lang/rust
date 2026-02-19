//@ build-pass
//@ only-aarch64
#![crate_type = "lib"]
#![allow(incomplete_features, internal_features)]
#![feature(abi_unadjusted, core_intrinsics, link_llvm_intrinsics, rustc_attrs)]

// Tests that tuples of scalable vectors are passed as immediates and that the intrinsics for
// creating/getting/setting tuples of scalable vectors generate the correct assembly

#[derive(Copy, Clone)]
#[rustc_scalable_vector(4)]
#[allow(non_camel_case_types)]
pub struct svfloat32_t(f32);

#[derive(Copy, Clone)]
#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
pub struct svfloat32x2_t(svfloat32_t, svfloat32_t);

#[derive(Copy, Clone)]
#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
pub struct svfloat32x3_t(svfloat32_t, svfloat32_t, svfloat32_t);

#[derive(Copy, Clone)]
#[rustc_scalable_vector]
#[allow(non_camel_case_types)]
pub struct svfloat32x4_t(svfloat32_t, svfloat32_t, svfloat32_t, svfloat32_t);

#[inline(never)]
#[target_feature(enable = "sve")]
pub fn svdup_n_f32(op: f32) -> svfloat32_t {
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.dup.x.nxv4f32")]
        fn _svdup_n_f32(op: f32) -> svfloat32_t;
    }
    unsafe { _svdup_n_f32(op) }
}

// CHECK: define { <vscale x 4 x float>, <vscale x 4 x float> } @svcreate2_f32(<vscale x 4 x float> %x0, <vscale x 4 x float> %x1)
#[no_mangle]
#[target_feature(enable = "sve")]
pub fn svcreate2_f32(x0: svfloat32_t, x1: svfloat32_t) -> svfloat32x2_t {
    // CHECK: %1 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float> } poison, <vscale x 4 x float> %x0, 0
    // CHECK-NEXT: %2 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float> } %1, <vscale x 4 x float> %x1, 1
    unsafe { std::intrinsics::simd::scalable::sve_tuple_create2(x0, x1) }
}

// CHECK: define { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @svcreate3_f32(<vscale x 4 x float> %x0, <vscale x 4 x float> %x1, <vscale x 4 x float> %x2)
#[no_mangle]
#[target_feature(enable = "sve")]
pub fn svcreate3_f32(x0: svfloat32_t, x1: svfloat32_t, x2: svfloat32_t) -> svfloat32x3_t {
    // CHECK-LABEL: @_RNvCsk3YxfLN8zWY_6tuples13svcreate3_f32
    // CHECK: %1 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } poison, <vscale x 4 x float> %x0, 0
    // CHECK-NEXT: %2 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } %1, <vscale x 4 x float> %x1, 1
    // CHECK-NEXT: %3 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } %2, <vscale x 4 x float> %x2, 2
    unsafe { std::intrinsics::simd::scalable::sve_tuple_create3(x0, x1, x2) }
}

// CHECK: define { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } @svcreate4_f32(<vscale x 4 x float> %x0, <vscale x 4 x float> %x1, <vscale x 4 x float> %x2, <vscale x 4 x float> %x3)
#[no_mangle]
#[target_feature(enable = "sve")]
pub fn svcreate4_f32(
    x0: svfloat32_t,
    x1: svfloat32_t,
    x2: svfloat32_t,
    x3: svfloat32_t,
) -> svfloat32x4_t {
    // CHECK-LABEL: @_RNvCsk3YxfLN8zWY_6tuples13svcreate4_f32
    // CHECK: %1 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } poison, <vscale x 4 x float> %x0, 0
    // CHECK-NEXT: %2 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } %1, <vscale x 4 x float> %x1, 1
    // CHECK-NEXT: %3 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } %2, <vscale x 4 x float> %x2, 2
    // CHECK-NEXT: %4 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float>, <vscale x 4 x float> } %3, <vscale x 4 x float> %x3, 3
    unsafe { std::intrinsics::simd::scalable::sve_tuple_create4(x0, x1, x2, x3) }
}

// CHECK: define <vscale x 4 x float> @svget2_f32({ <vscale x 4 x float>, <vscale x 4 x float> } %tup)
#[no_mangle]
#[target_feature(enable = "sve")]
pub fn svget2_f32<const IDX: i32>(tup: svfloat32x2_t) -> svfloat32_t {
    // CHECK: %1 = extractvalue { <vscale x 4 x float>, <vscale x 4 x float> } %tup, 0
    unsafe { std::intrinsics::simd::scalable::sve_tuple_get::<_, _, { IDX }>(tup) }
}

// CHECK: define { <vscale x 4 x float>, <vscale x 4 x float> } @svset2_f32({ <vscale x 4 x float>, <vscale x 4 x float> } %tup, <vscale x 4 x float> %x)
#[no_mangle]
#[target_feature(enable = "sve")]
pub fn svset2_f32<const IDX: i32>(tup: svfloat32x2_t, x: svfloat32_t) -> svfloat32x2_t {
    // CHECK: %1 = insertvalue { <vscale x 4 x float>, <vscale x 4 x float> } %tup, <vscale x 4 x float> %x, 0
    unsafe { std::intrinsics::simd::scalable::sve_tuple_set::<_, _, { IDX }>(tup, x) }
}

// This function exists only so there are calls to the generic functions
#[target_feature(enable = "sve")]
pub fn test() {
    let x = svdup_n_f32(2f32);
    let tup = svcreate2_f32(x, x);
    let x = svget2_f32::<0>(tup);
    let tup = svset2_f32::<0>(tup, x);
}
