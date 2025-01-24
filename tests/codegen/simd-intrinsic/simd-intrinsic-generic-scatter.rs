//

//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![feature(repr_simd, intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec2<T>(pub [T; 2]);

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vec4<T>(pub [T; 4]);

extern "rust-intrinsic" {
    fn simd_scatter<T, P, M>(value: T, pointers: P, mask: M);
}

// CHECK-LABEL: @scatter_f32x2
#[no_mangle]
pub unsafe fn scatter_f32x2(pointers: Vec2<*mut f32>, mask: Vec2<i32>,
                            values: Vec2<f32>) {
    // CHECK: call void @llvm.masked.scatter.v2f32.v2p0(<2 x float> {{.*}}, <2 x ptr> {{.*}}, i32 {{.*}}, <2 x i1> {{.*}})
    simd_scatter(values, pointers, mask)
}


// CHECK-LABEL: @scatter_pf32x2
#[no_mangle]
pub unsafe fn scatter_pf32x2(pointers: Vec2<*mut *const f32>, mask: Vec2<i32>,
                             values: Vec2<*const f32>) {
    // CHECK: call void @llvm.masked.scatter.v2p0.v2p0(<2 x ptr> {{.*}}, <2 x ptr> {{.*}}, i32 {{.*}}, <2 x i1> {{.*}})
    simd_scatter(values, pointers, mask)
}
