//@ compile-flags: -O -Z merge-functions=disabled --edition=2021
//@ only-x86_64
// FIXME: Remove the `min-llvm-version`.
//@ revisions: NINETEEN TWENTY
//@[NINETEEN] exact-llvm-major-version: 19
//@[TWENTY] min-llvm-version: 20
//@ min-llvm-version: 19

#![crate_type = "lib"]
#![feature(try_blocks)]

use std::ops::ControlFlow::{self, Break, Continue};
use std::ptr::NonNull;

// CHECK-LABEL: @option_nop_match_32
#[no_mangle]
pub fn option_nop_match_32(x: Option<u32>) -> Option<u32> {
    // CHECK: start:
    // TWENTY-NEXT: %trunc = trunc nuw i32 %0 to i1
    // TWENTY-NEXT: %.2 = select i1 %trunc, i32 %1, i32 undef
    // CHECK-NEXT: [[REG1:%.*]] = insertvalue { i32, i32 } poison, i32 %0, 0
    // NINETEEN-NEXT: [[REG2:%.*]] = insertvalue { i32, i32 } [[REG1]], i32 %1, 1
    // TWENTY-NEXT: [[REG2:%.*]] = insertvalue { i32, i32 } [[REG1]], i32 %.2, 1
    // CHECK-NEXT: ret { i32, i32 } [[REG2]]
    match x {
        Some(x) => Some(x),
        None => None,
    }
}

// CHECK-LABEL: @option_nop_traits_32
#[no_mangle]
pub fn option_nop_traits_32(x: Option<u32>) -> Option<u32> {
    // CHECK: start:
    // TWENTY-NEXT: %trunc = trunc nuw i32 %0 to i1
    // TWENTY-NEXT: %.1 = select i1 %trunc, i32 %1, i32 undef
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: ret { i32, i32 }
    try { x? }
}

// CHECK-LABEL: @result_nop_match_32
#[no_mangle]
pub fn result_nop_match_32(x: Result<i32, u32>) -> Result<i32, u32> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: ret { i32, i32 }
    match x {
        Ok(x) => Ok(x),
        Err(x) => Err(x),
    }
}

// CHECK-LABEL: @result_nop_traits_32
#[no_mangle]
pub fn result_nop_traits_32(x: Result<i32, u32>) -> Result<i32, u32> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: ret { i32, i32 }
    try { x? }
}

// CHECK-LABEL: @result_nop_match_64
#[no_mangle]
pub fn result_nop_match_64(x: Result<i64, u64>) -> Result<i64, u64> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: ret { i64, i64 }
    match x {
        Ok(x) => Ok(x),
        Err(x) => Err(x),
    }
}

// CHECK-LABEL: @result_nop_traits_64
#[no_mangle]
pub fn result_nop_traits_64(x: Result<i64, u64>) -> Result<i64, u64> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: ret { i64, i64 }
    try { x? }
}

// CHECK-LABEL: @result_nop_match_ptr
#[no_mangle]
pub fn result_nop_match_ptr(x: Result<usize, Box<()>>) -> Result<usize, Box<()>> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i{{[0-9]+}}, ptr }
    // CHECK-NEXT: insertvalue { i{{[0-9]+}}, ptr }
    // CHECK-NEXT: ret
    match x {
        Ok(x) => Ok(x),
        Err(x) => Err(x),
    }
}

// CHECK-LABEL: @result_nop_traits_ptr
#[no_mangle]
pub fn result_nop_traits_ptr(x: Result<u64, NonNull<()>>) -> Result<u64, NonNull<()>> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i{{[0-9]+}}, ptr }
    // CHECK-NEXT: insertvalue { i{{[0-9]+}}, ptr }
    // CHECK-NEXT: ret
    try { x? }
}

// CHECK-LABEL: @control_flow_nop_match_32
#[no_mangle]
pub fn control_flow_nop_match_32(x: ControlFlow<i32, u32>) -> ControlFlow<i32, u32> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: ret { i32, i32 }
    match x {
        Continue(x) => Continue(x),
        Break(x) => Break(x),
    }
}

// CHECK-LABEL: @control_flow_nop_traits_32
#[no_mangle]
pub fn control_flow_nop_traits_32(x: ControlFlow<i32, u32>) -> ControlFlow<i32, u32> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: insertvalue { i32, i32 }
    // CHECK-NEXT: ret { i32, i32 }
    try { x? }
}
