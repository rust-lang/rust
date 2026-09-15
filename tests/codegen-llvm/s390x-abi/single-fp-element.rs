//@ add-minicore
//@ needs-llvm-components: systemz
//@ compile-flags: --target=s390x-unknown-linux-gnu -Copt-level=3 -Zmerge-functions=disabled
#![crate_type = "lib"]
#![feature(no_core, f16, f128)]
#![no_core]

extern crate minicore;
use minicore::hint::black_box;
use minicore::*;

#[repr(C)]
struct Wrapper<T>(T);

// CHECK: define void @plain_f16(half noundef %x)
#[unsafe(no_mangle)]
extern "C" fn plain_f16(x: f16) {
    black_box(x);
}

// CHECK: define void @wrapped_f16(half %0)
#[unsafe(no_mangle)]
extern "C" fn wrapped_f16(x: Wrapper<f16>) {
    black_box(x);
}

// CHECK: define void @plain_f32(float noundef %x)
#[unsafe(no_mangle)]
extern "C" fn plain_f32(x: f32) {
    black_box(x);
}

// CHECK: define void @wrapped_f32(float %0)
#[unsafe(no_mangle)]
extern "C" fn wrapped_f32(x: Wrapper<f32>) {
    black_box(x);
}

// CHECK: define void @plain_f64(double noundef %x)
#[unsafe(no_mangle)]
extern "C" fn plain_f64(x: f64) {
    black_box(x);
}

// CHECK: define void @wrapped_f64(double %0)
#[unsafe(no_mangle)]
extern "C" fn wrapped_f64(x: Wrapper<f64>) {
    black_box(x);
}

// CHECK: define void @plain_f128(ptr {{.*}}dereferenceable(16) %x)
#[unsafe(no_mangle)]
extern "C" fn plain_f128(x: f128) {
    black_box(x);
}

// CHECK: define void @wrapped_f128(ptr {{.*}}dereferenceable(16) %x)
#[unsafe(no_mangle)]
extern "C" fn wrapped_f128(x: Wrapper<f128>) {
    black_box(x);
}

#[repr(transparent)]
struct Transparent<T>(T);

// CHECK: define void @transparent_wrapped_f32(float %0)
#[unsafe(no_mangle)]
extern "C" fn transparent_wrapped_f32(x: Transparent<Wrapper<f32>>) {
    black_box(x);
}

// CHECK: define void @transparent_transparent_wrapped_f32(float %0)
#[unsafe(no_mangle)]
extern "C" fn transparent_transparent_wrapped_f32(x: Transparent<Transparent<Wrapper<f32>>>) {
    black_box(x);
}

#[repr(C, align(8))]
struct Aligned8Wrapper<T>(T);

// CHECK: define void @aligned_8_wrapped_f16(double %0)
#[unsafe(no_mangle)]
extern "C" fn aligned_8_wrapped_f16(x: Aligned8Wrapper<f16>) {
    black_box(x);
}

// CHECK: define void @aligned_8_wrapped_f32(double %0)
#[unsafe(no_mangle)]
extern "C" fn aligned_8_wrapped_f32(x: Aligned8Wrapper<f32>) {
    black_box(x);
}

#[repr(C, align(16))]
struct Aligned16Wrapper<T>(T);

// CHECK: define void @aligned_16_wrapped_f32(ptr {{.*}}dereferenceable(16)
#[unsafe(no_mangle)]
extern "C" fn aligned_16_wrapped_f32(x: Aligned16Wrapper<f32>) {
    black_box(x);
}

#[repr(C)]
union UnionWrapper<T: Copy> {
    a: T,
}

// A repr(C) union does not count.
//
// CHECK: define void @union_wrapped_f32(i32 %0)
#[unsafe(no_mangle)]
extern "C" fn union_wrapped_f32(x: UnionWrapper<f32>) {
    black_box(x);
}

// But a repr(transparent) union does.
//
// CHECK: define void @maybe_uninit_f32(float %x)
#[unsafe(no_mangle)]
extern "C" fn maybe_uninit_f32(x: MaybeUninit<f32>) {
    black_box(x);
}

// A single-element array also does not count.
//
// CHECK: define void @array_f32(i32 %0)
#[unsafe(no_mangle)]
extern "C" fn array_f32(x: [f32; 1]) {
    black_box(x);
}
