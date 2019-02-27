// ignore-emscripten
// ignore-tidy-linelength

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![feature(repr_simd, platform_intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec2<T>(pub T, pub T);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec4<T>(pub T, pub T, pub T, pub T);

extern "platform-intrinsic" {
    fn simd_gather<T, P, M>(value: T, pointers: P, mask: M) -> T;
}

// CHECK-LABEL: @gather_f32x2
#[no_mangle]
pub unsafe fn gather_f32x2(pointers: Vec2<*const f32>, mask: Vec2<i32>,
                           values: Vec2<f32>) -> Vec2<f32> {
    // CHECK: call <2 x float> @llvm.masked.gather.v2f32.v2p0f32(<2 x float*> {{.*}}, i32 {{.*}}, <2 x i1> {{.*}}, <2 x float> {{.*}})
    simd_gather(values, pointers, mask)
}

// CHECK-LABEL: @gather_pf32x2
#[no_mangle]
pub unsafe fn gather_pf32x2(pointers: Vec2<*const *const f32>, mask: Vec2<i32>,
                           values: Vec2<*const f32>) -> Vec2<*const f32> {
    // CHECK: call <2 x float*> @llvm.masked.gather.v2p0f32.v2p0p0f32(<2 x float**> {{.*}}, i32 {{.*}}, <2 x i1> {{.*}}, <2 x float*> {{.*}})
    simd_gather(values, pointers, mask)
}
