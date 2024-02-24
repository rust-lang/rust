//@ compile-flags: -C no-prepopulate-passes

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
    fn simd_masked_store<M, P, T>(mask: M, pointer: P, values: T) -> ();
}

// CHECK-LABEL: @store_f32x2
#[no_mangle]
pub unsafe fn store_f32x2(mask: Vec2<i32>, pointer: *mut f32, values: Vec2<f32>) {
    // CHECK: call void @llvm.masked.store.v2f32.p0(<2 x float> {{.*}}, ptr {{.*}}, i32 4, <2 x i1> {{.*}})
    simd_masked_store(mask, pointer, values)
}

// CHECK-LABEL: @store_pf32x4
#[no_mangle]
pub unsafe fn store_pf32x4(mask: Vec4<i32>, pointer: *mut *const f32, values: Vec4<*const f32>) {
    // CHECK: call void @llvm.masked.store.v4p0.p0(<4 x ptr> {{.*}}, ptr {{.*}}, i32 {{.*}}, <4 x i1> {{.*}})
    simd_masked_store(mask, pointer, values)
}
