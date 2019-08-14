// ignore-tidy-linelength
// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#![allow(non_camel_case_types, incomplete_features)]
#![feature(repr_simd, platform_intrinsics, const_generics)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct S<const N: usize>([f32; N]);

// CHECK-LABEL: @build_array
#[no_mangle]
pub fn build_array(x: [f32; 4]) -> S<4> {
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{[0-9]+}}, i8* align 4 %3, i64 16, i1 false)
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %{{[0-9]+}}, i8* align 4 %6, i64 16, i1 false)
    S::<4>(x)
}
