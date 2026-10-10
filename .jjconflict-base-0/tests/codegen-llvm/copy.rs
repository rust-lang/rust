//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ only-64bit (so I don't need to worry about usize)

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// This deals in a count of elements, not bytes, so we need to multiply.
// Ensure we preserve UB from a count too high to be valid.
use std::intrinsics::copy;

// CHECK-LABEL: @copy_u16(
#[no_mangle]
pub unsafe fn copy_u16(src: *const u16, dst: *mut u16, count: usize) {
    // CHECK: [[BYTES:%.+]] = mul nuw nsw i64 2, %count
    // CHECK: call void @llvm.memmove.p0.p0.i64(ptr align 2 %dst, ptr align 2 %src, i64 [[BYTES]], i1 false)
    copy(src, dst, count)
}
