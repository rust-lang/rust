//
//@ compile-flags: -C no-prepopulate-passes
// LLVM IR isn't very portable and the one tested here depends on the ABI
// which is different between x86 (where we use SSE registers) and others.
// `x86-64` and `x86-32-sse2` are identical, but compiletest does not support
// taking the union of multiple `only` annotations.
//@ revisions: x86-64 x86-32-sse2 other
//@[x86-64] only-x86_64
//@[x86-32-sse2] only-rustc_abi-x86-sse2
//@[other] ignore-rustc_abi-x86-sse2
//@[other] ignore-x86_64

#![crate_type = "lib"]
#![allow(non_camel_case_types)]
#![feature(repr_simd, intrinsics)]

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
    // x86-32: ret <4 x float> %[[VAL:.+]]
    // x86-64: ret <4 x float> %[[VAL:.+]]
    // other: store <4 x float> %[[VAL:.+]], ptr %_0, align [[VECTOR_ALIGN]]
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
    // x86-32: ret <4 x float> %[[VAL:.+]]
    // x86-64: ret <4 x float> %[[VAL:.+]]
    // other: store <4 x float> %[[VAL:.+]], ptr %_0, align [[VECTOR_ALIGN]]
    unsafe { std::mem::transmute(x) }
}
