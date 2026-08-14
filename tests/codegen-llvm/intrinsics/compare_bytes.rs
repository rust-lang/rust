//@ revisions: INT32 INT16
//@ compile-flags: -Copt-level=3
//@ [INT32] ignore-16bit
//@ [INT16] only-16bit

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::compare_bytes;

#[no_mangle]
// CHECK-LABEL: @bytes_cmp(
pub unsafe fn bytes_cmp(a: *const u8, b: *const u8, n: usize) -> std::cmp::Ordering {
    // INT32: %[[RAW:.+]] = tail call i32 @memcmp(ptr %a, ptr %b, {{i32|i64}} %n)
    // INT16: %[[RAW:.+]] = tail call i16 @memcmp(ptr %a, ptr %b, i16 %n)

    // INT32: %[[ORD:.+]] = tail call i8 @llvm.scmp.i8.i32(i32 %[[RAW]], i32 0)
    // INT16: %[[ORD:.+]] = tail call i8 @llvm.scmp.i8.i16(i16 %[[RAW]], i16 0)

    // CHECK: ret i8 %[[ORD]]
    compare_bytes(a, b, n)
}

// Ensure that, even though there's an `scmp` emitted by the intrinsic,
// that doesn't end up pessiming checks against zero.
#[no_mangle]
// CHECK-LABEL: @bytes_eq(
pub unsafe fn bytes_eq(a: *const u8, b: *const u8, n: usize) -> bool {
    // CHECK: %[[RAW:.+]] = tail call {{.+}} @{{bcmp|memcmp}}(ptr %a, ptr %b, {{i16|i32|i64}} %n)
    // INT32: %[[B:.+]] = icmp eq i32 %[[RAW]], 0
    // INT16: %[[B:.+]] = icmp eq i16 %[[RAW]], 0
    // CHECK: ret i1 %[[B]]
    compare_bytes(a, b, n).is_eq()
}
