//@ compile-flags: -Copt-level=3
// Test that `pow` can use a faster implementation when `base` is a
// known power of two

#![crate_type = "lib"]

// 2 ** n == 2 ** (1 * n) == 1 << (1 * n)
// CHECK-LABEL: @pow2
#[no_mangle]
pub fn pow2(exp: u32) -> u32 {
    // CHECK: %[[OVERFLOW:.+]] = icmp ult i32 %exp, 32
    // CHECK: %[[POW:.+]] = shl nuw i32 1, %exp
    // CHECK: %[[RET:.+]] = select i1 %[[OVERFLOW]], i32 %[[POW]], i32 0
    // CHECK: ret i32 %[[RET]]
    2u32.pow(exp)
}

// 4 ** n == 2 ** (2 * n) == 1 << (2 * n)
// CHECK-LABEL: @pow4
#[no_mangle]
pub fn pow4(exp: u32) -> u32 {
    // CHECK: %[[ICMP1:.+]] = icmp slt i32 %exp, 0
    // CHECK: %[[SHIFT_AMOUNT:.+]] = shl i32 %exp, 1
    // CHECK: %[[ICMP2:.+]] = icmp ult i32 %[[SHIFT_AMOUNT]], 32
    // CHECK: %[[POW:.+]] = shl nuw i32 1, %[[SHIFT_AMOUNT]]
    // CHECK: %[[SEL:.+]] = select i1 %[[ICMP2]], i32 %[[POW]], i32 0
    // CHECK: %[[RET:.+]] = select i1 %[[ICMP1]], i32 0, i32 %[[SEL]]
    // CHECK: ret i32 %[[RET]]
    4u32.pow(exp)
}

// 16 ** n == 2 ** (4 * n) == 1 << (4 * n)
// CHECK-LABEL: @pow16
#[no_mangle]
pub fn pow16(exp: u32) -> u32 {
    // CHECK: %[[ICMP1:.+]] = icmp ugt i32 %exp, 1073741823
    // CHECK: %[[SHIFT_AMOUNT:.+]] = shl i32 %exp, 2
    // CHECK: %[[ICMP2:.+]] = icmp ult i32 %[[SHIFT_AMOUNT]], 32
    // CHECK: %[[POW:.+]] = shl nuw i32 1, %[[SHIFT_AMOUNT]]
    // CHECK: %[[SEL:.+]] = select i1 %[[ICMP2]], i32 %[[POW]], i32 0
    // CHECK: %[[RET:.+]] = select i1 %[[ICMP1]], i32 0, i32 %[[SEL]]
    // CHECK: ret i32 %[[RET]]
    16u32.pow(exp)
}

// (-2) ** n == (-2) ** (1 * n) == 1 << (1 * n)
// CHECK-LABEL: @pow_minus_2
#[no_mangle]
pub fn pow_minus_2(exp: u32) -> i32 {
    // CHECK: %[[OVERFLOW:.+]] = icmp ult i32 %exp, 32
    // CHECK: %[[POW:.+]] = shl nuw i32 1, %exp
    // CHECK: %[[MAGNITUDE:.+]] = select i1 %[[OVERFLOW]], i32 %[[POW]], i32 0
    // CHECK: %[[IS_ODD:.+]] = and i32 %exp, 1
    // CHECK: %[[IS_EVEN:.+]] = icmp eq i32 %[[IS_ODD]], 0
    // CHECK: %[[NEG:.+]] = sub i32 0, %[[MAGNITUDE]]
    // CHECK: %[[RET:.+]] = select i1 %[[IS_EVEN]], i32 %[[MAGNITUDE]], i32 %[[NEG]]
    // CHECK: ret i32 %[[RET]]
    (-2i32).pow(exp)
}
