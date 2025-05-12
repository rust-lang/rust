//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

// This runs mir-opts to inline the standard library call, but doesn't run LLVM
// optimizations so it doesn't need to worry about them adding more flags.

#![crate_type = "lib"]
#![feature(unchecked_shifts)]
#![feature(core_intrinsics)]

// CHECK-LABEL: @unchecked_shl_unsigned_same
#[no_mangle]
pub unsafe fn unchecked_shl_unsigned_same(a: u32, b: u32) -> u32 {
    // CHECK-NOT: assume
    // CHECK-NOT: and i32
    // CHECK: shl i32 %a, %b
    // CHECK-NOT: and i32
    a.unchecked_shl(b)
}

// CHECK-LABEL: @unchecked_shl_unsigned_smaller
#[no_mangle]
pub unsafe fn unchecked_shl_unsigned_smaller(a: u16, b: u32) -> u16 {
    // CHECK-NOT: assume
    // CHECK: %[[TRUNC:.+]] = trunc nuw i32 %b to i16
    // CHECK: shl i16 %a, %[[TRUNC]]
    a.unchecked_shl(b)
}

// CHECK-LABEL: @unchecked_shl_unsigned_bigger
#[no_mangle]
pub unsafe fn unchecked_shl_unsigned_bigger(a: u64, b: u32) -> u64 {
    // CHECK-NOT: assume
    // CHECK: %[[EXT:.+]] = zext i32 %b to i64
    // CHECK: shl i64 %a, %[[EXT]]
    a.unchecked_shl(b)
}

// CHECK-LABEL: @unchecked_shr_signed_same
#[no_mangle]
pub unsafe fn unchecked_shr_signed_same(a: i32, b: u32) -> i32 {
    // CHECK-NOT: assume
    // CHECK-NOT: and i32
    // CHECK: ashr i32 %a, %b
    // CHECK-NOT: and i32
    a.unchecked_shr(b)
}

// CHECK-LABEL: @unchecked_shr_signed_smaller
#[no_mangle]
pub unsafe fn unchecked_shr_signed_smaller(a: i16, b: u32) -> i16 {
    // CHECK-NOT: assume
    // CHECK: %[[TRUNC:.+]] = trunc nuw i32 %b to i16
    // CHECK: ashr i16 %a, %[[TRUNC]]
    a.unchecked_shr(b)
}

// CHECK-LABEL: @unchecked_shr_signed_bigger
#[no_mangle]
pub unsafe fn unchecked_shr_signed_bigger(a: i64, b: u32) -> i64 {
    // CHECK-NOT: assume
    // CHECK: %[[EXT:.+]] = zext i32 %b to i64
    // CHECK: ashr i64 %a, %[[EXT]]
    a.unchecked_shr(b)
}

// CHECK-LABEL: @unchecked_shr_u128_i8
#[no_mangle]
pub unsafe fn unchecked_shr_u128_i8(a: u128, b: i8) -> u128 {
    // CHECK-NOT: assume
    // CHECK: %[[EXT:.+]] = zext i8 %b to i128
    // CHECK: lshr i128 %a, %[[EXT]]
    std::intrinsics::unchecked_shr(a, b)
}

// CHECK-LABEL: @unchecked_shl_i128_u8
#[no_mangle]
pub unsafe fn unchecked_shl_i128_u8(a: i128, b: u8) -> i128 {
    // CHECK-NOT: assume
    // CHECK: %[[EXT:.+]] = zext i8 %b to i128
    // CHECK: shl i128 %a, %[[EXT]]
    std::intrinsics::unchecked_shl(a, b)
}

// CHECK-LABEL: @unchecked_shl_u8_i128
#[no_mangle]
pub unsafe fn unchecked_shl_u8_i128(a: u8, b: i128) -> u8 {
    // CHECK-NOT: assume
    // CHECK: %[[TRUNC:.+]] = trunc nuw i128 %b to i8
    // CHECK: shl i8 %a, %[[TRUNC]]
    std::intrinsics::unchecked_shl(a, b)
}

// CHECK-LABEL: @unchecked_shr_i8_u128
#[no_mangle]
pub unsafe fn unchecked_shr_i8_u128(a: i8, b: u128) -> i8 {
    // CHECK-NOT: assume
    // CHECK: %[[TRUNC:.+]] = trunc nuw i128 %b to i8
    // CHECK: ashr i8 %a, %[[TRUNC]]
    std::intrinsics::unchecked_shr(a, b)
}
