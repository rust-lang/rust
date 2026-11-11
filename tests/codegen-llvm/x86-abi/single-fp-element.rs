//@ add-minicore
//@ needs-llvm-components: x86
//@ revisions: win linux
//@[win] compile-flags: --target i686-pc-windows-gnu
//@[linux] compile-flags: --target i686-unknown-linux-gnu -Zreg-struct-return=true
//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
#![crate_type = "lib"]
#![feature(no_core, f16, f128)]
#![no_core]

extern crate minicore;
use minicore::hint::black_box;
use minicore::*;

#[repr(C)]
struct Wrapper<T>(T);

// CHECK-LABEL: define noundef half @plain_f16(
#[unsafe(no_mangle)]
extern "C" fn plain_f16(x: f16) -> f16 {
    x
}

// CHECK-LABEL: define half @wrapped_f16(
#[unsafe(no_mangle)]
extern "C" fn wrapped_f16(x: Wrapper<f16>) -> Wrapper<f16> {
    x
}

// CHECK-LABEL: define noundef float @plain_f32(
#[unsafe(no_mangle)]
extern "C" fn plain_f32(x: f32) -> f32 {
    x
}

// CHECK-LABEL: define float @wrapped_f32(
#[unsafe(no_mangle)]
extern "C" fn wrapped_f32(x: Wrapper<f32>) -> Wrapper<f32> {
    x
}

// CHECK-LABEL: define noundef double @plain_f64(
#[unsafe(no_mangle)]
extern "C" fn plain_f64(x: f64) -> f64 {
    x
}

// CHECK-LABEL: define double @wrapped_f64(
#[unsafe(no_mangle)]
extern "C" fn wrapped_f64(x: Wrapper<f64>) -> Wrapper<f64> {
    x
}

// CHECK-LABEL: define noundef fp128 @plain_f128(
#[unsafe(no_mangle)]
extern "C" fn plain_f128(x: f128) -> f128 {
    x
}

// CHECK-LABEL: define void @wrapped_f128(ptr {{.*}}sret([16 x i8])
#[unsafe(no_mangle)]
extern "C" fn wrapped_f128(x: Wrapper<f128>) -> Wrapper<f128> {
    x
}

#[repr(transparent)]
struct Transparent<T>(T);

// CHECK-LABEL: define float @transparent_wrapped_f32(
#[unsafe(no_mangle)]
extern "C" fn transparent_wrapped_f32(x: Transparent<Wrapper<f32>>) -> Transparent<Wrapper<f32>> {
    x
}

// CHECK-LABEL: define float @transparent_transparent_wrapped_f32(
#[unsafe(no_mangle)]
extern "C" fn transparent_transparent_wrapped_f32(
    x: Transparent<Transparent<Wrapper<f32>>>,
) -> Transparent<Transparent<Wrapper<f32>>> {
    x
}

#[repr(align(4))]
struct Empty {}

#[repr(C)]
struct Struct {
    f: f32,
    a: [i32; 0],
    b: Empty,
}

// One or more aligned ZSTs are fine and do not disqualify the type.
// CHECK-LABEL: define float @aligned_zst_f32(
#[unsafe(no_mangle)]
extern "C" fn aligned_zst_f32(x: Struct) -> Struct {
    x
}

#[repr(C, align(8))]
struct AlignedWrapper<T>(T);

// Over-aligning disqualifies the type.
//
// CHECK-LABEL: define i64 @aligned_wrapped_f16(
#[unsafe(no_mangle)]
extern "C" fn aligned_wrapped_f16(x: AlignedWrapper<f16>) -> AlignedWrapper<f16> {
    x
}

// Over-aligning disqualifies the type.
//
// CHECK-LABEL: define i64 @aligned_wrapped_f32(
#[unsafe(no_mangle)]
extern "C" fn aligned_wrapped_f32(x: AlignedWrapper<f32>) -> AlignedWrapper<f32> {
    x
}

#[repr(C)]
union UnionWrapper<T: Copy> {
    a: T,
}

// A repr(C) union does count.
//
// CHECK-LABEL: define float @union_wrapped_f32(
#[unsafe(no_mangle)]
extern "C" fn union_wrapped_f32(x: UnionWrapper<f32>) -> UnionWrapper<f32> {
    x
}

// A repr(transparent) union does too.
//
// CHECK-LABEL: define float @maybe_uninit_f32(
#[unsafe(no_mangle)]
extern "C" fn maybe_uninit_f32(x: MaybeUninit<f32>) -> MaybeUninit<f32> {
    x
}

// A single-element array also does count.
//
// CHECK-LABEL: define float @array_f32(
#[unsafe(no_mangle)]
extern "C" fn array_f32(x: [f32; 1]) -> [f32; 1] {
    x
}
