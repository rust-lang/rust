//@ compile-flags: -Copt-level=3

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
    // This uses -DAG to avoid failing on irrelevant reorderings,
    // like emitting the truncation earlier.

    // CHECK-DAG: %[[INRANGE:.+]] = icmp ult i32 %b, 16
    // CHECK-DAG: tail call void @llvm.assume(i1 %[[INRANGE]])
    // CHECK-DAG: %[[TRUNC:.+]] = trunc{{( nuw)?( nsw)?}} i32 %b to i16
    // CHECK-DAG: shl i16 %a, %[[TRUNC]]
    a.unchecked_shl(b)
}

// CHECK-LABEL: @unchecked_shl_unsigned_bigger
#[no_mangle]
pub unsafe fn unchecked_shl_unsigned_bigger(a: u64, b: u32) -> u64 {
    // CHECK-NOT: assume
    // CHECK: %[[EXT:.+]] = zext{{( nneg)?}} i32 %b to i64
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
    // This uses -DAG to avoid failing on irrelevant reorderings,
    // like emitting the truncation earlier.

    // CHECK-DAG: %[[INRANGE:.+]] = icmp ult i32 %b, 16
    // CHECK-DAG: tail call void @llvm.assume(i1 %[[INRANGE]])
    // CHECK-DAG: %[[TRUNC:.+]] = trunc{{( nuw)?( nsw)?}}  i32 %b to i16
    // CHECK-DAG: ashr i16 %a, %[[TRUNC]]
    a.unchecked_shr(b)
}

// CHECK-LABEL: @unchecked_shr_signed_bigger
#[no_mangle]
pub unsafe fn unchecked_shr_signed_bigger(a: i64, b: u32) -> i64 {
    // CHECK-NOT: assume
    // CHECK: %[[EXT:.+]] = zext{{( nneg)?}} i32 %b to i64
    // CHECK: ashr i64 %a, %[[EXT]]
    a.unchecked_shr(b)
}

// CHECK-LABEL: @unchecked_shr_u128_i8
#[no_mangle]
pub unsafe fn unchecked_shr_u128_i8(a: u128, b: i8) -> u128 {
    // CHECK-NOT: assume
    // CHECK: %[[EXT:.+]] = zext{{( nneg)?}} i8 %b to i128
    // CHECK: lshr i128 %a, %[[EXT]]
    std::intrinsics::unchecked_shr(a, b)
}

// CHECK-LABEL: @unchecked_shl_i128_u8
#[no_mangle]
pub unsafe fn unchecked_shl_i128_u8(a: i128, b: u8) -> i128 {
    // CHECK-NOT: assume
    // CHECK: %[[EXT:.+]] = zext{{( nneg)?}} i8 %b to i128
    // CHECK: shl i128 %a, %[[EXT]]
    std::intrinsics::unchecked_shl(a, b)
}

// CHECK-LABEL: @unchecked_shl_u8_i128
#[no_mangle]
pub unsafe fn unchecked_shl_u8_i128(a: u8, b: i128) -> u8 {
    // This uses -DAG to avoid failing on irrelevant reorderings,
    // like emitting the truncation earlier.

    // CHECK-DAG: %[[INRANGE:.+]] = icmp ult i128 %b, 8
    // CHECK-DAG: tail call void @llvm.assume(i1 %[[INRANGE]])
    // CHECK-DAG: %[[TRUNC:.+]] = trunc{{( nuw)?( nsw)?}} i128 %b to i8
    // CHECK-DAG: shl i8 %a, %[[TRUNC]]
    std::intrinsics::unchecked_shl(a, b)
}

// CHECK-LABEL: @unchecked_shr_i8_u128
#[no_mangle]
pub unsafe fn unchecked_shr_i8_u128(a: i8, b: u128) -> i8 {
    // This uses -DAG to avoid failing on irrelevant reorderings,
    // like emitting the truncation earlier.

    // CHECK-DAG: %[[INRANGE:.+]] = icmp ult i128 %b, 8
    // CHECK-DAG: tail call void @llvm.assume(i1 %[[INRANGE]])
    // CHECK-DAG: %[[TRUNC:.+]] = trunc{{( nuw)?( nsw)?}} i128 %b to i8
    // CHECK-DAG: ashr i8 %a, %[[TRUNC]]
    std::intrinsics::unchecked_shr(a, b)
}
