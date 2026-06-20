//@ compile-flags: -Copt-level=3
// Test that `checked_ilog` can use a faster implementation when `base` is a
// known power of two

#![crate_type = "lib"]

// CHECK-LABEL: @checked_ilog2
#[no_mangle]
pub fn checked_ilog2(val: u32) -> Option<u32> {
    // CHECK: %[[ICMP:.+]] = icmp ne i32 %val, 0
    // CHECK: %[[CTZ:.+]] = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 %val, i1 true)
    // CHECK: %[[LOG2:.+]] = xor i32 %[[CTZ]], 31
    val.checked_ilog(2)
}

// log(4, x) == log(2, x) / 2
// CHECK-LABEL: @checked_ilog4
#[no_mangle]
pub fn checked_ilog4(val: u32) -> Option<u32> {
    // CHECK: %[[ICMP:.+]] = icmp ne i32 %val, 0
    // CHECK: %[[CTZ:.+]] = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 %val, i1 true)
    // CHECK: %[[DIV2:.+]] = lshr i32 %[[CTZ]], 1
    // CHECK: %[[LOG4:.+]] = xor i32 %[[DIV2]], 15
    val.checked_ilog(4)
}

// log(16, x) == log(2, x) / 4
// CHECK-LABEL: @checked_ilog16
#[no_mangle]
pub fn checked_ilog16(val: u32) -> Option<u32> {
    // CHECK: %[[ICMP:.+]] = icmp ne i32 %val, 0
    // CHECK: %[[CTZ:.+]] = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 %val, i1 true)
    // CHECK: %[[DIV4:.+]] = lshr i32 %[[CTZ]], 2
    // CHECK: %[[LOG16:.+]] = xor i32 %[[DIV2]], 7
    val.checked_ilog(16)
}
