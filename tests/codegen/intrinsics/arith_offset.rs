//@ compile-flags: -O

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::arith_offset;

// CHECK-LABEL: ptr @arith_offset_zst
// CHECK-SAME: (ptr noundef{{.*}} %p, [[SIZE:i[0-9]+]] noundef %d)
#[no_mangle]
pub unsafe fn arith_offset_zst(p: *const (), d: isize) -> *const () {
    // CHECK-NOT: getelementptr
    // CHECK: ret ptr %p
    arith_offset(p, d)
}

// CHECK-LABEL: ptr @arith_offset_u32
// CHECK-SAME: (ptr noundef{{.*}} %p, [[SIZE]] noundef %d)
#[no_mangle]
pub unsafe fn arith_offset_u32(p: *const u32, d: isize) -> *const u32 {
    // CHECK: %[[R:.*]] = getelementptr [4 x i8], ptr %p, [[SIZE]] %d
    // CHECK-NEXT: ret ptr %[[R]]
    arith_offset(p, d)
}

// CHECK-LABEL: ptr @arith_offset_u64
// CHECK-SAME: (ptr noundef{{.*}} %p, [[SIZE]] noundef %d)
#[no_mangle]
pub unsafe fn arith_offset_u64(p: *const u64, d: isize) -> *const u64 {
    // CHECK: %[[R:.*]] = getelementptr [8 x i8], ptr %p, [[SIZE]] %d
    // CHECK-NEXT: ret ptr %[[R]]
    arith_offset(p, d)
}
