//@ compile-flags: -Cno-prepopulate-passes

#![crate_type = "lib"]
#![feature(repr_simd, core_intrinsics)]
// make sure that codegen emits correctly-aligned loads and stores for repr(packed, simd) types
// the alignment of a load should be no less than T, and no more than the size of the vector type
use std::intrinsics::simd as intrinsics;

#[derive(Copy, Clone)]
#[repr(packed, simd)]
struct f32x3([f32; 3]);

#[derive(Copy, Clone)]
#[repr(packed, simd)]
struct f32x4([f32; 4]);

// CHECK-LABEL: load_f32x3
#[no_mangle]
pub fn load_f32x3(floats: &f32x3) -> f32x3 {
    // FIXME: Is a memcpy really the best we can do?
    // CHECK: @llvm.memcpy.{{.*}}ptr align 4 {{.*}}ptr align 4
    *floats
}

// CHECK-LABEL: load_f32x4
#[no_mangle]
pub fn load_f32x4(floats: &f32x4) -> f32x4 {
    // CHECK: load <4 x float>, ptr %{{[a-z0-9_]*}}, align {{4|8|16}}
    *floats
}

// CHECK-LABEL: add_f32x3
#[no_mangle]
pub fn add_f32x3(x: f32x3, y: f32x3) -> f32x3 {
    // CHECK: load <3 x float>, ptr %{{[a-z0-9_]*}}, align 4
    unsafe { intrinsics::simd_add(x, y) }
}

// CHECK-LABEL: add_f32x4
#[no_mangle]
pub fn add_f32x4(x: f32x4, y: f32x4) -> f32x4 {
    // CHECK: load <4 x float>, ptr %{{[a-z0-9_]*}}, align {{4|8|16}}
    unsafe { intrinsics::simd_add(x, y) }
}
