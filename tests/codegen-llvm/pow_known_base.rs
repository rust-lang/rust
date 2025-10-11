//@ compile-flags: -Copt-level=3
// Test that `pow` can use a faster implementation when `base` is a
// known power of two

#![crate_type = "lib"]

// CHECK-LABEL: @pow2
#[no_mangle]
pub fn pow2(exp: u32) -> u32 {
    // CHECK: %[[SHIFT_AMOUNT:.+]] = and i32 %exp, 31
    // CHECK: %[[POW2:.+]] = shl nuw i32 1, %[[SHIFT_AMOUNT]]
    // CHECK: ret i32 %[[POW2]]
    2u32.pow(exp)
}

// 4 ** n == 2 ** (2 * n) == 1 << (2 * n)
// CHECK-LABEL: @pow4
#[no_mangle]
pub fn pow4(exp: u32) -> u32 {
    // CHECK: %[[EXP2:.+]] = shl i32 %exp, 1
    // CHECK: %[[SHIFT_AMOUNT:.+]] = and i32 %[[EXP2]], 30
    // CHECK: %[[POW4:.+]] = shl nuw nsw i32 1, %[[SHIFT_AMOUNT]]
    // CHECK: ret i32 %[[POW4]]
    4u32.pow(exp)
}

// 16 ** n == 2 ** (4 * n) == 1 << (4 * n)
// CHECK-LABEL: @pow16
#[no_mangle]
pub fn pow16(exp: u32) -> u32 {
    // CHECK: %[[EXP2:.+]] = shl i32 %exp, 2
    // CHECK: %[[SHIFT_AMOUNT:.+]] = and i32 %[[EXP2]], 28
    // CHECK: %[[POW16:.+]] = shl nuw nsw i32 1, %[[SHIFT_AMOUNT]]
    // CHECK: ret i32 %[[POW16]]
    16u32.pow(exp)
}
