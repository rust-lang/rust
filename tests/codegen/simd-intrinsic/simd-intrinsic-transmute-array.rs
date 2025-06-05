//
//@ compile-flags: -C no-prepopulate-passes
// 32bit MSVC does not align things properly so we suppress high alignment annotations (#112480)
//@ ignore-i686-pc-windows-msvc
//@ ignore-i686-pc-windows-gnu

#![crate_type = "lib"]
#![allow(non_camel_case_types)]
#![feature(repr_simd, core_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct S<const N: usize>([f32; N]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct T([f32; 4]);

// CHECK-LABEL: @array_align(
#[no_mangle]
pub fn array_align() -> usize {
    // CHECK: ret [[USIZE:i[0-9]+]] [[ARRAY_ALIGN:[0-9]+]]
    const { std::mem::align_of::<f32>() }
}

// CHECK-LABEL: @vector_align(
#[no_mangle]
pub fn vector_align() -> usize {
    // CHECK: ret [[USIZE]] [[VECTOR_ALIGN:[0-9]+]]
    const { std::mem::align_of::<T>() }
}

// CHECK-LABEL: @build_array_s
#[no_mangle]
pub fn build_array_s(x: [f32; 4]) -> S<4> {
    // CHECK: call void @llvm.memcpy.{{.+}}({{.*}} align [[VECTOR_ALIGN]] {{.*}} align [[ARRAY_ALIGN]] {{.*}}, [[USIZE]] 16, i1 false)
    S::<4>(x)
}

// CHECK-LABEL: @build_array_transmute_s
#[no_mangle]
pub fn build_array_transmute_s(x: [f32; 4]) -> S<4> {
    // CHECK: %[[VAL:.+]] = load <4 x float>, ptr %x, align [[ARRAY_ALIGN]]
    // CHECK: store <4 x float> %[[VAL:.+]], ptr %_0, align [[VECTOR_ALIGN]]
    unsafe { std::mem::transmute(x) }
}

// CHECK-LABEL: @build_array_t
#[no_mangle]
pub fn build_array_t(x: [f32; 4]) -> T {
    // CHECK: call void @llvm.memcpy.{{.+}}({{.*}} align [[VECTOR_ALIGN]] {{.*}} align [[ARRAY_ALIGN]] {{.*}}, [[USIZE]] 16, i1 false)
    T(x)
}

// CHECK-LABEL: @build_array_transmute_t
#[no_mangle]
pub fn build_array_transmute_t(x: [f32; 4]) -> T {
    // CHECK: %[[VAL:.+]] = load <4 x float>, ptr %x, align [[ARRAY_ALIGN]]
    // CHECK: store <4 x float> %[[VAL:.+]], ptr %_0, align [[VECTOR_ALIGN]]
    unsafe { std::mem::transmute(x) }
}
