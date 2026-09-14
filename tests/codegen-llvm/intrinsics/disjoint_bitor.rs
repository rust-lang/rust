//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::disjoint_bitor;

// CHECK-LABEL: @disjoint_bitor_signed
#[no_mangle]
pub unsafe fn disjoint_bitor_signed(x: i32, y: i32) -> i32 {
    // CHECK: [[TEMP:%.+]] = or disjoint i32 %x, %y
    // CHECK: ret i32 [[TEMP]]
    disjoint_bitor(x, y)
}

// CHECK-LABEL: @disjoint_bitor_unsigned
#[no_mangle]
pub unsafe fn disjoint_bitor_unsigned(x: u64, y: u64) -> u64 {
    // CHECK: [[TEMP:%.+]] = or disjoint i64 %x, %y
    // CHECK: ret i64 [[TEMP]]
    disjoint_bitor(x, y)
}

// CHECK-LABEL: @disjoint_bitor_literal
#[no_mangle]
pub unsafe fn disjoint_bitor_literal() -> u8 {
    // This is a separate check because even without any passes,
    // LLVM will fold so it's not an instruction, which can assert in LLVM.

    // CHECK: ret i8 3
    disjoint_bitor(1, 2)
}
