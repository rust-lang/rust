//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::offset;

// CHECK-LABEL: ptr @offset_zst
// CHECK-SAME: (ptr noundef %p, [[SIZE:i[0-9]+]] noundef %d)
#[no_mangle]
pub unsafe fn offset_zst(p: *const (), d: usize) -> *const () {
    // CHECK-NOT: getelementptr
    // CHECK: ret ptr %p
    offset(p, d)
}

// CHECK-LABEL: ptr @offset_isize
// CHECK-SAME: (ptr noundef %p, [[SIZE]] noundef %d)
#[no_mangle]
pub unsafe fn offset_isize(p: *const u32, d: isize) -> *const u32 {
    // CHECK: %[[R:.*]] = getelementptr inbounds i32, ptr %p, [[SIZE]] %d
    // CHECK-NEXT: ret ptr %[[R]]
    offset(p, d)
}

// CHECK-LABEL: ptr @offset_usize
// CHECK-SAME: (ptr noundef %p, [[SIZE]] noundef %d)
#[no_mangle]
pub unsafe fn offset_usize(p: *const u64, d: usize) -> *const u64 {
    // CHECK: %[[R:.*]] = getelementptr inbounds{{( nuw)?}} i64, ptr %p, [[SIZE]] %d
    // CHECK-NEXT: ret ptr %[[R]]
    offset(p, d)
}
