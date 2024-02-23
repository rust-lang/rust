//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![feature(repr_simd, intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec2<T>(pub T, pub T);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec4<T>(pub T, pub T, pub T, pub T);

extern "rust-intrinsic" {
    fn simd_masked_load<M, P, T>(mask: M, pointer: P, values: T) -> T;
}

// CHECK-LABEL: @load_f32x2
#[no_mangle]
pub unsafe fn load_f32x2(mask: Vec2<i32>, pointer: *const f32,
                         values: Vec2<f32>) -> Vec2<f32> {
    // CHECK: call <2 x float> @llvm.masked.load.v2f32.p0(ptr {{.*}}, i32 4, <2 x i1> {{.*}}, <2 x float> {{.*}})
    simd_masked_load(mask, pointer, values)
}

// CHECK-LABEL: @load_pf32x4
#[no_mangle]
pub unsafe fn load_pf32x4(mask: Vec4<i32>, pointer: *const *const f32,
                          values: Vec4<*const f32>) -> Vec4<*const f32> {
    // CHECK: call <4 x ptr> @llvm.masked.load.v4p0.p0(ptr {{.*}}, i32 {{.*}}, <4 x i1> {{.*}}, <4 x ptr> {{.*}})
    simd_masked_load(mask, pointer, values)
}
