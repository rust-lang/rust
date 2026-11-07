//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Z inline-mir
//@ only-64bit (so I don't need to worry about usize)

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::ptr_metadata;

// CHECK-LABEL: @thin_metadata(
#[no_mangle]
pub fn thin_metadata(p: *const ()) {
    // CHECK: start
    // CHECK-NEXT: ret void
    ptr_metadata(p)
}

// CHECK-LABEL: @slice_metadata(
#[no_mangle]
pub fn slice_metadata(p: *const [u8]) -> usize {
    // CHECK: start
    // CHECK-NEXT: ret i64 %p.1
    ptr_metadata(p)
}

// CHECK-LABEL: @dyn_byte_offset(
#[no_mangle]
pub unsafe fn dyn_byte_offset(
    p: *const dyn std::fmt::Debug,
    n: usize,
) -> *const dyn std::fmt::Debug {
    // CHECK: %[[Q:.+]] = getelementptr inbounds{{( nuw)?}} i8, ptr %p.0, i64 %n
    // CHECK: %[[TEMP1:.+]] = insertvalue { ptr, ptr } poison, ptr %[[Q]], 0
    // CHECK: %[[TEMP2:.+]] = insertvalue { ptr, ptr } %[[TEMP1]], ptr %p.1, 1
    // CHECK: ret { ptr, ptr } %[[TEMP2]]
    p.byte_add(n)
}
