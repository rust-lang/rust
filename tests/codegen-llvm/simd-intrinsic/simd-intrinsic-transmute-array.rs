//
//@ compile-flags: -C no-prepopulate-passes
// 32bit MSVC does not align things properly so we suppress high alignment annotations (#112480)
//@ ignore-i686-pc-windows-msvc
//@ ignore-i686-pc-windows-gnu

// LLVM IR isn't very portable and the one tested here depends on the ABI which is different between
// x86 (where we use SSE registers) and others. `x86-64` and `x86-32-sse2` are identical, but
// compiletest does not support taking the union of multiple `only` annotations.
//@ revisions: x86-64 x86-32-sse2 by-ref
//@[x86-64] only-x86_64
//@[x86-64] filecheck-flags: --check-prefix=by-val
//@[x86-32-sse2] only-rustc_abi-x86-sse2
//@[x86-32-sse2] filecheck-flags: --check-prefix=by-val
//@[by-ref] ignore-rustc_abi-x86-sse2
//@[by-ref] ignore-x86_64

#![crate_type = "lib"]
#![allow(non_camel_case_types)]
#![feature(repr_simd, core_intrinsics)]

#[path = "../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

pub type S<const N: usize> = Simd<f32, N>;

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
    const { std::mem::align_of::<S<4>>() }
}

// CHECK-LABEL: @build_array
#[no_mangle]
pub fn build_array(x: [f32; 4]) -> S<4> {
    // by-val: %[[VAL:.+]] = load <4 x float>, ptr %x, align [[ARRAY_ALIGN]]
    // by-val: ret <4 x float> %[[VAL:.+]]
    // by-ref: call void @llvm.memcpy.{{.+}}({{.*}} align [[VECTOR_ALIGN]] {{.*}} align [[ARRAY_ALIGN]] {{.*}}, [[USIZE]] 16, i1 false)
    Simd(x)
}

// CHECK-LABEL: @build_array_transmute
#[no_mangle]
pub fn build_array_transmute(x: [f32; 4]) -> S<4> {
    // by-val: %[[VAL:.+]] = load <4 x float>, ptr %x, align [[ARRAY_ALIGN]]
    // by-val: ret <4 x float> %[[VAL:.+]]
    // by-ref: call void @llvm.memcpy.{{.+}}({{.*}} align [[VECTOR_ALIGN]] {{.*}} align [[ARRAY_ALIGN]] {{.*}}, [[USIZE]] 16, i1 false)
    unsafe { std::mem::transmute(x) }
}
