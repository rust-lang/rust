//@ revisions: INT32 INT16
//@ compile-flags: -Copt-level=3
//@ [INT32] ignore-16bit
//@ [INT16] only-16bit

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::compare_bytes;

#[no_mangle]
// CHECK-LABEL: @bytes_cmp(
pub unsafe fn bytes_cmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    // INT32: %[[TEMP:.+]] = tail call i32 @memcmp(ptr %a, ptr %b, {{i32|i64}} %n)
    // INT32-NOT: sext
    // INT32: ret i32 %[[TEMP]]

    // INT16: %[[TEMP1:.+]] = tail call i16 @memcmp(ptr %a, ptr %b, i16 %n)
    // INT16: %[[TEMP2:.+]] = sext i16 %[[TEMP1]] to i32
    // INT16: ret i32 %[[TEMP2]]
    compare_bytes(a, b, n)
}

// Ensure that, even though there's an `sext` emitted by the intrinsic,
// that doesn't end up pessiming checks against zero.
#[no_mangle]
// CHECK-LABEL: @bytes_eq(
pub unsafe fn bytes_eq(a: *const u8, b: *const u8, n: usize) -> bool {
    // CHECK: call {{.+}} @{{bcmp|memcmp}}(ptr %a, ptr %b, {{i16|i32|i64}} %n)
    // CHECK-NOT: sext
    // INT32: icmp eq i32
    // INT16: icmp eq i16
    compare_bytes(a, b, n) == 0_i32
}
