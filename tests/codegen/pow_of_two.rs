// compile-flags: --crate-type=lib

// CHECK: @b = unnamed_addr alias i64 (i32), ptr @a
// CHECK-LABEL: @a(
#[no_mangle]
pub fn a(exp: u32) -> u64 {
    // CHECK: %[[R:.+]] = and i32 %exp, 63
    // CHECK: %[[R:.+]] = zext i32 %[[R:.+]] to i64
    // CHECK: %[[R:.+]].i = shl nuw i64 1, %[[R:.+]]
    // CHECK: ret i64 %[[R:.+]].i
    2u64.pow(exp)
}

#[no_mangle]
pub fn b(exp: u32) -> i64 {
    2i64.pow(exp)
}

// CHECK-LABEL: @c(
#[no_mangle]
pub fn c(exp: u32) -> u32 {
    // CHECK: %[[R:.+]].0.i = shl i32 %exp, 1
    // CHECK: %[[R:.+]].1.i = icmp sgt i32 %exp, -1
    // CHECK: %[[R:.+]] = and i32 %[[R:.+]].0.i, 30
    // CHECK: %[[R:.+]].i = zext i1 %[[R:.+]].1.i to i32
    // CHECK: %[[R:.+]] = shl nuw nsw i32 %[[R:.+]].i, %[[R:.+]]
    // CHECK: ret i32 %[[R:.+]]
    4u32.pow(exp)
}

// CHECK-LABEL: @d(
#[no_mangle]
pub fn d(exp: u32) -> u32 {
    // CHECK: %[[R:.+]] = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %exp, i32 5)
    // CHECK: %[[R:.+]].0.i = extractvalue { i32, i1 } %[[R:.+]], 0
    // CHECK: %[[R:.+]].1.i = extractvalue { i32, i1 } %[[R:.+]], 1
    // CHECK: %[[R:.+]].i = xor i1 %[[R:.+]].1.i, true
    // CHECK: %[[R:.+]] = and i32 %[[R:.+]].0.i, 31
    // CHECK: %[[R:.+]].i = zext i1 %[[R:.+]].i to i32
    // CHECK: %[[R:.+]] = shl nuw i32 %[[R:.+]].i, %[[R:.+]]
    // CHECK: ret i32 %[[R:.+]]
    32u32.pow(exp)
}

// CHECK-LABEL: @e(
#[no_mangle]
pub fn e(exp: u32) -> i32 {
    // CHECK: %[[R:.+]] = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %exp, i32 5)
    // CHECK: %[[R:.+]].1.i = extractvalue { i32, i1 } %[[R:.+]], 1
    // CHECK: %[[R:.+]].i = xor i1 %[[R:.+]].1.i, true
    // CHECK: %[[R:.+]].i = zext i1 %[[R:.+]].i to i32
    // CHECK: %[[R:.+]].0.i = extractvalue { i32, i1 } %[[R:.+]], 0
    // CHECK: %[[R:.+]] = and i32 %[[R:.+]].0.i, 31
    // CHECK: %[[R:.+]].010.i = shl nuw i32 %[[R:.+]].i, %[[R:.+]]
    // CHECK: ret i32 %[[R:.+]].010.i
    32i32.pow(exp)
}
