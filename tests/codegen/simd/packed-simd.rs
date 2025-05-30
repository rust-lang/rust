//@ revisions:opt3 noopt
//@ only-x86_64
//@[opt3] compile-flags: -Copt-level=3
//@[noopt] compile-flags: -Cno-prepopulate-passes

#![crate_type = "lib"]
#![no_std]
#![feature(repr_simd, core_intrinsics)]
use core::intrinsics::simd as intrinsics;
use core::{mem, ptr};

// Test codegen for not only "packed" but also "fully aligned" SIMD types, and conversion between
// them. A repr(packed,simd) type with 3 elements can't exceed its element alignment, whereas the
// same type as repr(simd) will instead have padding.

#[repr(simd, packed)]
#[derive(Copy, Clone)]
pub struct PackedSimd<T, const N: usize>([T; N]);

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct FullSimd<T, const N: usize>([T; N]);

// non-powers-of-two have padding and need to be expanded to full vectors
fn load<T, const N: usize>(v: PackedSimd<T, N>) -> FullSimd<T, N> {
    unsafe {
        let mut tmp = mem::MaybeUninit::<FullSimd<T, N>>::uninit();
        ptr::copy_nonoverlapping(&v as *const _, tmp.as_mut_ptr().cast(), 1);
        tmp.assume_init()
    }
}

// CHECK-LABEL: square_packed_full
// CHECK-SAME: ptr{{[a-z_ ]*}} sret([[RET_TYPE:[^)]+]]) [[RET_ALIGN:align (8|16)]]{{[^%]*}} [[RET_VREG:%[_0-9]*]]
// CHECK-SAME: ptr{{[a-z_ ]*}} align 4
#[no_mangle]
pub fn square_packed_full(x: PackedSimd<f32, 3>) -> FullSimd<f32, 3> {
    // CHECK-NEXT: start
    // noopt: alloca [[RET_TYPE]], [[RET_ALIGN]]
    // CHECK: load <3 x float>
    let x = load(x);
    // CHECK: [[VREG:%[a-z0-9_]+]] = fmul <3 x float>
    // CHECK-NEXT: store <3 x float> [[VREG]], ptr [[RET_VREG]], [[RET_ALIGN]]
    // CHECK-NEXT: ret void
    unsafe { intrinsics::simd_mul(x, x) }
}

// CHECK-LABEL: square_packed
// CHECK-SAME: ptr{{[a-z_ ]*}} sret([[RET_TYPE:[^)]+]]) [[RET_ALIGN:align 4]]{{[^%]*}} [[RET_VREG:%[_0-9]*]]
// CHECK-SAME: ptr{{[a-z_ ]*}} align 4
#[no_mangle]
pub fn square_packed(x: PackedSimd<f32, 3>) -> PackedSimd<f32, 3> {
    // CHECK-NEXT: start
    // CHECK-NEXT: load <3 x float>
    // noopt-NEXT: load <3 x float>
    // CHECK-NEXT: [[VREG:%[a-z0-9_]+]] = fmul <3 x float>
    // CHECK-NEXT: store <3 x float> [[VREG]], ptr [[RET_VREG]], [[RET_ALIGN]]
    // CHECK-NEXT: ret void
    unsafe { intrinsics::simd_mul(x, x) }
}
