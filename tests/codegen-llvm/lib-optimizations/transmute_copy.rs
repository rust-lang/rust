//@ compile-flags: -Copt-level=1
//@ only-64bit

#![crate_type = "lib"]

// Check that we don't have runtime alignment checks, just a safe alignment.
// (Rust emits a branch, but LLVM merges the loads from the arms.)

use std::mem::transmute_copy;

// CHECK-LABEL: @transmute_copy_i16_from_i32_slice
// CHECK-SAME: ptr{{.+}}%_0,
// CHECK-SAME: ptr{{.+}}%x.0,
// CHECK-SAME: i64{{.+}}%x.1)
#[no_mangle]
pub unsafe fn transmute_copy_i16_from_i32_slice(x: &[i32]) -> [i16; 7] {
    // CHECK: start:
    // CHECK-NEXT: @llvm.memcpy
    // CHECK-SAME: align 2{{.+}}%_0,
    // CHECK-SAME: align 4{{.+}}%x.0,
    // CHECK-SAME: i64 14,
    // CHECK-NEXT: ret void
    transmute_copy(x)
}

// CHECK-LABEL: @transmute_copy_i32_from_i16_slice
// CHECK-SAME: ptr{{.+}}%_0,
// CHECK-SAME: ptr{{.+}}%x.0,
// CHECK-SAME: i64{{.+}}%x.1)
#[no_mangle]
pub unsafe fn transmute_copy_i32_from_i16_slice(x: &[i16]) -> [i32; 3] {
    // CHECK: start:
    // CHECK-NEXT: @llvm.memcpy
    // CHECK-SAME: align 4{{.+}}%_0,
    // CHECK-SAME: align 2{{.+}}%x.0,
    // CHECK-SAME: i64 12,
    // CHECK-NEXT: ret void
    transmute_copy(x)
}

// CHECK-LABEL: i32 @transmute_copy_i32_from_dyn
// CHECK-SAME: ptr{{.+}}%x.0,
// CHECK-SAME: ptr{{.+}}%x.1)
#[no_mangle]
pub unsafe fn transmute_copy_i32_from_dyn(x: &dyn std::fmt::Debug) -> i32 {
    // CHECK: start:
    // CHECK-NEXT: [[TEMP:%.+]] = load i32, ptr %x.0, align 1
    // CHECK-NEXT: ret i32 [[TEMP]]
    transmute_copy(x)
}

// CHECK-LABEL: @transmute_copy_i16_array_from_dyn
// CHECK-SAME: ptr{{.+}}%_0,
// CHECK-SAME: ptr{{.+}}%x.0,
// CHECK-SAME: ptr{{.+}}%x.1)
#[no_mangle]
pub unsafe fn transmute_copy_i16_array_from_dyn(x: &dyn std::fmt::Debug) -> [i16; 7] {
    // CHECK: start:
    // CHECK-NEXT: @llvm.memcpy
    // CHECK-SAME: align 2{{.+}}%_0,
    // CHECK-SAME: align 1{{.+}}%x.0,
    // CHECK-SAME: i64 14,
    // CHECK-NEXT: ret void
    transmute_copy(x)
}
