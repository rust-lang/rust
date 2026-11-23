//@ assembly-output: emit-asm
//@ add-minicore
//@ revisions: linux windows-gnu windows-msvc
//@[linux] compile-flags: --target i686-unknown-linux-gnu
//@[linux] needs-llvm-components: x86
//@[windows-gnu] compile-flags: --target i686-pc-windows-gnu
//@[windows-gnu] needs-llvm-components: x86
//@[windows-msvc] compile-flags: --target i686-pc-windows-msvc
//@[windows-msvc] needs-llvm-components: x86
// We want to test LLVM optimisations, so disabled MIR optimisations.
//@ compile-flags: -Copt-level=3 -Zmir-opt-level=0 -Zmerge-functions=disabled

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: noundef x86_fp80 @return_f32(float noundef %x)
// CHECK: start:
// FIXME(llvm22): The hexadecimal can be removed once LLVM 22 is dropped.
// CHECK-NEXT: %0 = fcmp uno float %x, {{0\.000000e\+00|0xK00000000000000000000}}
// CHECK-NEXT: br i1 %0, label %float_pre_ret.is_nan, label %float_pre_ret.is_not_nan
// CHECK: float_pre_ret.is_nan:
// CHECK-NEXT: %1 = bitcast float %x to i32
// CHECK-NEXT: %2 = lshr i32 %1, 16
// CHECK-NEXT: %3 = or i32 %2, 32767
// CHECK-NEXT: %4 = zext nneg i32 %3 to i80
// CHECK-NEXT: %5 = shl nuw i80 %4, 64
// CHECK-NEXT: %6 = zext i32 %1 to i64
// CHECK-NEXT: %7 = shl i64 %6, 40
// CHECK-NEXT: %8 = zext i64 %7 to i80
// CHECK-NEXT: %9 = or disjoint i80 %5, %8
// CHECK-NEXT: %10 = bitcast i80 %9 to x86_fp80
// CHECK-NEXT: br label %float_pre_ret.after
// CHECK: float_pre_ret.is_not_nan:
// CHECK-NEXT: %11 = fpext float %x to x86_fp80
// CHECK-NEXT: br label %float_pre_ret.after
// CHECK: float_pre_ret.after:
// CHECK-NEXT: %12 = phi x86_fp80 [ %10, %float_pre_ret.is_nan ], [ %11, %float_pre_ret.is_not_nan ]
// CHECK-NEXT: ret x86_fp80 %12
#[unsafe(no_mangle)]
extern "C" fn return_f32(x: f32) -> f32 {
    x
}

// CHECK-LABEL: noundef x86_fp80 @return_f64(double noundef %x)
// CHECK: start:
// FIXME(llvm22): The hexadecimal can be removed once LLVM 22 is dropped.
// CHECK-NEXT: %0 = fcmp uno double %x, {{0\.000000e\+00|0xK00000000000000000000}}
// CHECK-NEXT: br i1 %0, label %float_pre_ret.is_nan, label %float_pre_ret.is_not_nan
// CHECK: float_pre_ret.is_nan:
// CHECK-NEXT: %1 = bitcast double %x to i64
// CHECK-NEXT: %2 = lshr i64 %1, 48
// CHECK-NEXT: %3 = or i64 %2, 32767
// CHECK-NEXT: %4 = zext nneg i64 %3 to i80
// CHECK-NEXT: %5 = shl nuw i80 %4, 64
// CHECK-NEXT: %6 = shl i64 %1, 11
// CHECK-NEXT: %7 = zext i64 %6 to i80
// CHECK-NEXT: %8 = or disjoint i80 %5, %7
// CHECK-NEXT: %9 = bitcast i80 %8 to x86_fp80
// CHECK-NEXT: br label %float_pre_ret.after
// CHECK: float_pre_ret.is_not_nan:
// CHECK-NEXT: %10 = fpext double %x to x86_fp80
// CHECK-NEXT: br label %float_pre_ret.after
// CHECK: float_pre_ret.after:
// CHECK-NEXT: %11 = phi x86_fp80 [ %9, %float_pre_ret.is_nan ], [ %10, %float_pre_ret.is_not_nan ]
// CHECK-NEXT: ret x86_fp80 %11
#[unsafe(no_mangle)]
extern "C" fn return_f64(x: f64) -> f64 {
    x
}

unsafe extern "C" {
    safe fn external_f32() -> f32;
    safe fn external_f64() -> f64;
}

// CHECK-LABEL: @call_f32(ptr {{.*}} %x)
// CHECK: start:
// CHECK-NEXT: %0 = {{.*}} call noundef x86_fp80 @external_f32()
// FIXME(llvm22): The hexadecimal can be removed once LLVM 22 is dropped.
// CHECK-NEXT: %1 = fcmp uno x86_fp80 %0, {{0\.000000e\+00|0xK00000000000000000000}}
// CHECK-NEXT: br i1 %1, label %float_post_ret.is_nan, label %float_post_ret.is_not_nan
// CHECK: float_post_ret.is_nan:
// CHECK-NEXT: %2 = bitcast x86_fp80 %0 to i80
// CHECK-NEXT: %3 = lshr i80 %2, 48
// CHECK-NEXT: %4 = trunc nuw i80 %3 to i32
// CHECK-NEXT: %5 = and i32 %4, -8388608
// CHECK-NEXT: %6 = trunc i80 %2 to i64
// CHECK-NEXT: %7 = lshr i64 %6, 40
// CHECK-NEXT: %8 = trunc nuw nsw i64 %7 to i32
// CHECK-NEXT: %9 = or i32 %5, %8
// CHECK-NEXT: %10 = bitcast i32 %9 to float
// CHECK-NEXT: br label %float_post_ret.after
// CHECK: float_post_ret.is_not_nan:
// CHECK-NEXT: %11 = fptrunc x86_fp80 %0 to float
// CHECK-NEXT: br label %float_post_ret.after
// CHECK: float_post_ret.after:
// CHECK-NEXT: %12 = phi float [ %10, %float_post_ret.is_nan ], [ %11, %float_post_ret.is_not_nan ]
// CHECK-NEXT: store float %12, ptr %x
// CHECK-NEXT: ret void
#[unsafe(no_mangle)]
extern "C" fn call_f32(x: &mut f32) {
    *x = external_f32();
}

// CHECK-LABEL: @call_f64(ptr {{.*}} %x)
// CHECK: start:
// CHECK-NEXT: %0 = {{.*}} call noundef x86_fp80 @external_f64()
// FIXME(llvm22): The hexadecimal can be removed once LLVM 22 is dropped.
// CHECK-NEXT: %1 = fcmp uno x86_fp80 %0, {{0\.000000e\+00|0xK00000000000000000000}}
// CHECK-NEXT: br i1 %1, label %float_post_ret.is_nan, label %float_post_ret.is_not_nan
// CHECK: float_post_ret.is_nan:
// CHECK-NEXT: %2 = bitcast x86_fp80 %0 to i80
// CHECK-NEXT: %3 = lshr i80 %2, 16
// CHECK-NEXT: %4 = trunc nuw i80 %3 to i64
// CHECK-NEXT: %5 = and i64 %4, -4503599627370496
// CHECK-NEXT: %6 = trunc i80 %2 to i64
// CHECK-NEXT: %7 = lshr i64 %6, 11
// CHECK-NEXT: %8 = or i64 %5, %7
// CHECK-NEXT: %9 = bitcast i64 %8 to double
// CHECK-NEXT: br label %float_post_ret.after
// CHECK: float_post_ret.is_not_nan:
// CHECK-NEXT: %10 = fptrunc x86_fp80 %0 to double
// CHECK-NEXT: br label %float_post_ret.after
// CHECK: float_post_ret.after:
// CHECK-NEXT: %11 = phi double [ %9, %float_post_ret.is_nan ], [ %10, %float_post_ret.is_not_nan ]
// CHECK-NEXT: store double %11, ptr %x
// CHECK-NEXT: ret void
#[unsafe(no_mangle)]
extern "C" fn call_f64(x: &mut f64) {
    *x = external_f64();
}

// Check undef values don't cause undefined behaviour.

// CHECK-LABEL: @return_undef_f32()
// CHECK-NOT: unreachable
// CHECK: ret x86_fp80
#[unsafe(no_mangle)]
extern "C" fn return_undef_f32() -> MaybeUninit<f32> {
    MaybeUninit::uninit()
}

// CHECK-LABEL: @return_undef_f64()
// CHECK-NOT: unreachable
// CHECK: ret x86_fp80
#[unsafe(no_mangle)]
extern "C" fn return_undef_f64() -> MaybeUninit<f64> {
    MaybeUninit::uninit()
}

// CHECK-LABEL: @call_undef_f32()
// CHECK-NOT: unreachable
// CHECK: ret x86_fp80
#[unsafe(no_mangle)]
extern "C" fn call_undef_f32() -> MaybeUninit<f32> {
    return_undef_f32()
}

// CHECK-LABEL: @call_undef_f64()
// CHECK-NOT: unreachable
// CHECK: ret x86_fp80
#[unsafe(no_mangle)]
extern "C" fn call_undef_f64() -> MaybeUninit<f64> {
    return_undef_f64()
}

// Check LLVM can still propogate constants

// CHECK-LABEL: @return_constant_f32()
// FIXME(llvm22): The hexadecimal can be removed once LLVM 22 is dropped.
// CHECK: ret x86_fp80 {{1\.500000e\+00|0xK3FFFC000000000000000}}
#[unsafe(no_mangle)]
extern "C" fn return_constant_f32() -> f32 {
    1.5
}

// CHECK-LABEL: @return_constant_f64()
// FIXME(llvm22): The hexadecimal can be removed once LLVM 22 is dropped.
// CHECK: ret x86_fp80 {{2\.500000e\+00|0xK4000A000000000000000}}
#[unsafe(no_mangle)]
extern "C" fn return_constant_f64() -> f64 {
    2.5
}

// CHECK-LABEL: @call_constant_f32()
// FIXME(llvm22): The hexadecimal can be removed once LLVM 22 is dropped.
// CHECK: ret x86_fp80 {{1\.500000e\+00|0xK3FFFC000000000000000}}
#[unsafe(no_mangle)]
extern "C" fn call_constant_f32() -> f32 {
    return_constant_f32()
}

// CHECK-LABEL: @call_constant_f64()
// FIXME(llvm22): The hexadecimal can be removed once LLVM 22 is dropped.
// CHECK: ret x86_fp80 {{2\.500000e\+00|0xK4000A000000000000000}}
#[unsafe(no_mangle)]
extern "C" fn call_constant_f64() -> f64 {
    return_constant_f64()
}

// Check that LLVM can optimise away the NaN branch when the value is guaranteed to be non-NaN.

// CHECK-LABEL: @return_non_nan_f32(i16 {{.*}} %x)
// CHECK: %0 = sitofp i16 %x to x86_fp80
// CHECK-NEXT: ret x86_fp80 %0
#[unsafe(no_mangle)]
extern "C" fn return_non_nan_f32(x: i16) -> f32 {
    x as _
}

// CHECK-LABEL: @return_non_nan_f64(i32 {{.*}} %x)
// CHECK: %0 = sitofp i32 %x to x86_fp80
// CHECK-NEXT: ret x86_fp80 %0
#[unsafe(no_mangle)]
extern "C" fn return_non_nan_f64(x: i32) -> f64 {
    x as _
}

// CHECK-LABEL: @call_non_nan_f32(i16 {{.*}} %x)
// CHECK: %0 = sitofp i16 %x to x86_fp80
// CHECK-NEXT: ret x86_fp80 %0
#[unsafe(no_mangle)]
extern "C" fn call_non_nan_f32(x: i16) -> f32 {
    return_non_nan_f32(x)
}

// CHECK-LABEL: @call_non_nan_f64(i32 {{.*}} %x)
// CHECK: %0 = sitofp i32 %x to x86_fp80
// CHECK-NEXT: ret x86_fp80 %0
#[unsafe(no_mangle)]
extern "C" fn call_non_nan_f64(x: i32) -> f64 {
    return_non_nan_f64(x)
}
