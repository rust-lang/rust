//@ revisions: z10 z10_vector z13 z13_no_vector
// ignore-tidy-linelength
//@ assembly-output: emit-asm
//@ compile-flags: -O -Z merge-functions=disabled
//@[z10] compile-flags: --target s390x-unknown-linux-gnu --cfg no_vector
//@[z10] needs-llvm-components: systemz
//@[z10_vector] compile-flags: --target s390x-unknown-linux-gnu -C target-feature=+vector
//@[z10_vector] needs-llvm-components: systemz
//@[z13] compile-flags: --target s390x-unknown-linux-gnu -C target-cpu=z13
//@[z13] needs-llvm-components: systemz
//@[z13_no_vector] compile-flags: --target s390x-unknown-linux-gnu -C target-cpu=z13 -C target-feature=-vector --cfg no_vector
//@[z13_no_vector] needs-llvm-components: systemz

#![feature(no_core, lang_items, repr_simd, s390x_target_feature)]
#![no_core]
#![crate_type = "lib"]
#![allow(non_camel_case_types)]

// Cases where vector feature is disabled are rejected.
// See tests/ui/simd-abi-checks-s390x.rs for test for them.

#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}
#[lang = "freeze"]
pub trait Freeze {}

impl<T: Copy, const N: usize> Copy for [T; N] {}

#[repr(simd)]
pub struct i8x8([i8; 8]);
#[repr(simd)]
pub struct i8x16([i8; 16]);
#[repr(simd)]
pub struct i8x32([i8; 32]);
#[repr(C)]
pub struct Wrapper<T>(T);
#[repr(transparent)]
pub struct TransparentWrapper<T>(T);

impl Copy for i8 {}
impl Copy for i64 {}
impl Copy for i8x8 {}
impl Copy for i8x16 {}
impl Copy for i8x32 {}
impl<T: Copy> Copy for Wrapper<T> {}
impl<T: Copy> Copy for TransparentWrapper<T> {}

// CHECK-LABEL: vector_ret_small:
// CHECK: vlrepg %v24, 0(%r2)
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_ret_small(x: &i8x8) -> i8x8 {
    *x
}
// CHECK-LABEL: vector_ret:
// CHECK: vl %v24, 0(%r2), 3
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_ret(x: &i8x16) -> i8x16 {
    *x
}
// CHECK-LABEL: vector_ret_large:
// z10: vl %v0, 16(%r3), 4
// z10-NEXT: vl %v1, 0(%r3), 4
// z10-NEXT: vst %v0, 16(%r2), 4
// z10-NEXT: vst %v1, 0(%r2), 4
// z10-NEXT: br %r14
// z13: vl %v0, 0(%r3), 4
// z13-NEXT: vl %v1, 16(%r3), 4
// z13-NEXT: vst %v1, 16(%r2), 4
// z13-NEXT: vst %v0, 0(%r2), 4
// z13-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_ret_large(x: &i8x32) -> i8x32 {
    *x
}

// CHECK-LABEL: vector_wrapper_ret_small:
// CHECK: mvc 0(8,%r2), 0(%r3)
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_wrapper_ret_small(x: &Wrapper<i8x8>) -> Wrapper<i8x8> {
    *x
}
// CHECK-LABEL: vector_wrapper_ret:
// CHECK: mvc 0(16,%r2), 0(%r3)
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_wrapper_ret(x: &Wrapper<i8x16>) -> Wrapper<i8x16> {
    *x
}
// CHECK-LABEL: vector_wrapper_ret_large:
// z10: vl %v0, 16(%r3), 4
// z10-NEXT: vl %v1, 0(%r3), 4
// z10-NEXT: vst %v0, 16(%r2), 4
// z10-NEXT: vst %v1, 0(%r2), 4
// z10-NEXT: br %r14
// z13: vl %v0, 16(%r3), 4
// z13-NEXT: vst %v0, 16(%r2), 4
// z13-NEXT: vl %v0, 0(%r3), 4
// z13-NEXT: vst %v0, 0(%r2), 4
// z13-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_wrapper_ret_large(x: &Wrapper<i8x32>) -> Wrapper<i8x32> {
    *x
}

// CHECK-LABEL: vector_transparent_wrapper_ret_small:
// CHECK: vlrepg %v24, 0(%r2)
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_transparent_wrapper_ret_small(
    x: &TransparentWrapper<i8x8>,
) -> TransparentWrapper<i8x8> {
    *x
}
// CHECK-LABEL: vector_transparent_wrapper_ret:
// CHECK: vl %v24, 0(%r2), 3
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_transparent_wrapper_ret(
    x: &TransparentWrapper<i8x16>,
) -> TransparentWrapper<i8x16> {
    *x
}
// CHECK-LABEL: vector_transparent_wrapper_ret_large:
// z10: vl %v0, 16(%r3), 4
// z10-NEXT: vl %v1, 0(%r3), 4
// z10-NEXT: vst %v0, 16(%r2), 4
// z10-NEXT: vst %v1, 0(%r2), 4
// z10-NEXT: br %r14
// z13: vl %v0, 0(%r3), 4
// z13-NEXT: vl %v1, 16(%r3), 4
// z13-NEXT: vst %v1, 16(%r2), 4
// z13-NEXT: vst %v0, 0(%r2), 4
// z13-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_transparent_wrapper_ret_large(
    x: &TransparentWrapper<i8x32>,
) -> TransparentWrapper<i8x32> {
    *x
}

// CHECK-LABEL: vector_arg_small:
// CHECK: vlgvg %r2, %v24, 0
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_arg_small(x: i8x8) -> i64 {
    unsafe { *(&x as *const i8x8 as *const i64) }
}
// CHECK-LABEL: vector_arg:
// CHECK: vlgvg %r2, %v24, 0
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_arg(x: i8x16) -> i64 {
    unsafe { *(&x as *const i8x16 as *const i64) }
}
// CHECK-LABEL: vector_arg_large:
// CHECK: lg %r2, 0(%r2)
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_arg_large(x: i8x32) -> i64 {
    unsafe { *(&x as *const i8x32 as *const i64) }
}

// CHECK-LABEL: vector_wrapper_arg_small:
// CHECK: vlgvg %r2, %v24, 0
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_wrapper_arg_small(x: Wrapper<i8x8>) -> i64 {
    unsafe { *(&x as *const Wrapper<i8x8> as *const i64) }
}
// CHECK-LABEL: vector_wrapper_arg:
// CHECK: vlgvg %r2, %v24, 0
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_wrapper_arg(x: Wrapper<i8x16>) -> i64 {
    unsafe { *(&x as *const Wrapper<i8x16> as *const i64) }
}
// CHECK-LABEL: vector_wrapper_arg_large:
// CHECK: lg %r2, 0(%r2)
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_wrapper_arg_large(x: Wrapper<i8x32>) -> i64 {
    unsafe { *(&x as *const Wrapper<i8x32> as *const i64) }
}

// CHECK-LABEL: vector_transparent_wrapper_arg_small:
// CHECK: vlgvg %r2, %v24, 0
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_transparent_wrapper_arg_small(x: TransparentWrapper<i8x8>) -> i64 {
    unsafe { *(&x as *const TransparentWrapper<i8x8> as *const i64) }
}
// CHECK-LABEL: vector_transparent_wrapper_arg:
// CHECK: vlgvg %r2, %v24, 0
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_transparent_wrapper_arg(x: TransparentWrapper<i8x16>) -> i64 {
    unsafe { *(&x as *const TransparentWrapper<i8x16> as *const i64) }
}
// CHECK-LABEL: vector_transparent_wrapper_arg_large:
// CHECK: lg %r2, 0(%r2)
// CHECK-NEXT: br %r14
#[cfg_attr(no_vector, target_feature(enable = "vector"))]
#[no_mangle]
unsafe extern "C" fn vector_transparent_wrapper_arg_large(x: TransparentWrapper<i8x32>) -> i64 {
    unsafe { *(&x as *const TransparentWrapper<i8x32> as *const i64) }
}
