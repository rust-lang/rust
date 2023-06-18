// compile-flags: -O
// min-llvm-version: 15.0 (LLVM 13 in CI does this differently from submodule LLVM)
// ignore-debug (because unchecked is checked in debug)

#![crate_type = "lib"]
#![feature(unchecked_math)]

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

    // CHECK-DAG: %[[INRANGE:.+]] = icmp ult i32 %b, 65536
    // CHECK-DAG: tail call void @llvm.assume(i1 %[[INRANGE]])
    // CHECK-DAG: %[[TRUNC:.+]] = trunc i32 %b to i16
    // CHECK-DAG: shl i16 %a, %[[TRUNC]]
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
    // This uses -DAG to avoid failing on irrelevant reorderings,
    // like emitting the truncation earlier.

    // CHECK-DAG: %[[INRANGE:.+]] = icmp ult i32 %b, 32768
    // CHECK-DAG: tail call void @llvm.assume(i1 %[[INRANGE]])
    // CHECK-DAG: %[[TRUNC:.+]] = trunc i32 %b to i16
    // CHECK-DAG: ashr i16 %a, %[[TRUNC]]
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
