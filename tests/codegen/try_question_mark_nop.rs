//@ compile-flags: -Copt-level=3 -Z merge-functions=disabled
//@ edition: 2021
//@ only-x86_64
//@ revisions: NINETEEN TWENTY
//@[NINETEEN] exact-llvm-major-version: 19
//@[TWENTY] min-llvm-version: 20

#![crate_type = "lib"]
#![feature(try_blocks)]

use std::ops::ControlFlow::{self, Break, Continue};
use std::ptr::NonNull;

// CHECK-LABEL: @option_nop_match_32
#[no_mangle]
pub fn option_nop_match_32(x: Option<u32>) -> Option<u32> {
    // CHECK: start:
    // CHECK-NEXT: [[TRUNC:%.*]] = trunc nuw i32 %0 to i1

    // NINETEEN-NEXT: [[SELECT:%.*]] = select i1 [[TRUNC]], i32 %0, i32 0
    // NINETEEN-NEXT: [[REG2:%.*]] = insertvalue { i32, i32 } poison, i32 [[SELECT]], 0
    // NINETEEN-NEXT: [[REG3:%.*]] = insertvalue { i32, i32 } [[REG2]], i32 %1, 1

    // TWENTY-NEXT: [[SELECT:%.*]] = select i1 [[TRUNC]], i32 %1, i32 undef
    // TWENTY-NEXT: [[REG2:%.*]] = insertvalue { i32, i32 } poison, i32 %0, 0
    // TWENTY-NEXT: [[REG3:%.*]] = insertvalue { i32, i32 } [[REG2]], i32 [[SELECT]], 1

    // CHECK-NEXT: ret { i32, i32 } [[REG3]]
    match x {
        Some(x) => Some(x),
        None => None,
    }
}

// CHECK-LABEL: @option_nop_traits_32
#[no_mangle]
pub fn option_nop_traits_32(x: Option<u32>) -> Option<u32> {
    // CHECK: start:
    // TWENTY-NEXT: %[[IS_SOME:.+]] = trunc nuw i32 %0 to i1
    // TWENTY-NEXT: select i1 %[[IS_SOME]], i32 %1, i32 undef
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

// CHECK-LABEL: @option_nop_match_64
#[no_mangle]
pub fn option_nop_match_64(x: Option<u64>) -> Option<u64> {
    // CHECK: start:
    // CHECK-NEXT: [[TRUNC:%.*]] = trunc nuw i64 %0 to i1

    // NINETEEN-NEXT: [[SELECT:%.*]] = select i1 [[TRUNC]], i64 %0, i64 0
    // NINETEEN-NEXT: [[REG2:%.*]] = insertvalue { i64, i64 } poison, i64 [[SELECT]], 0
    // NINETEEN-NEXT: [[REG3:%.*]] = insertvalue { i64, i64 } [[REG2]], i64 %1, 1

    // TWENTY-NEXT: [[SELECT:%.*]] = select i1 [[TRUNC]], i64 %1, i64 undef
    // TWENTY-NEXT: [[REG2:%.*]] = insertvalue { i64, i64 } poison, i64 %0, 0
    // TWENTY-NEXT: [[REG3:%.*]] = insertvalue { i64, i64 } [[REG2]], i64 [[SELECT]], 1

    // CHECK-NEXT: ret { i64, i64 } [[REG3]]
    match x {
        Some(x) => Some(x),
        None => None,
    }
}

// CHECK-LABEL: @option_nop_traits_64
#[no_mangle]
pub fn option_nop_traits_64(x: Option<u64>) -> Option<u64> {
    // CHECK: start:
    // TWENTY-NEXT: %[[TRUNC:[0-9]+]] = trunc nuw i64 %0 to i1
    // TWENTY-NEXT: %[[SEL:\.[0-9]+]] = select i1 %[[TRUNC]], i64 %1, i64 undef
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: ret { i64, i64 }
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

// CHECK-LABEL: @control_flow_nop_match_64
#[no_mangle]
pub fn control_flow_nop_match_64(x: ControlFlow<i64, u64>) -> ControlFlow<i64, u64> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: ret { i64, i64 }
    match x {
        Continue(x) => Continue(x),
        Break(x) => Break(x),
    }
}

// CHECK-LABEL: @control_flow_nop_traits_64
#[no_mangle]
pub fn control_flow_nop_traits_64(x: ControlFlow<i64, u64>) -> ControlFlow<i64, u64> {
    // CHECK: start:
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: insertvalue { i64, i64 }
    // CHECK-NEXT: ret { i64, i64 }
    try { x? }
}

// CHECK-LABEL: @result_nop_match_128
#[no_mangle]
pub fn result_nop_match_128(x: Result<i128, u128>) -> Result<i128, u128> {
    // CHECK: start:
    // CHECK-NEXT: store i128
    // CHECK-NEXT: getelementptr inbounds {{(nuw )?}}i8
    // CHECK-NEXT: store i128
    // CHECK-NEXT: ret void
    match x {
        Ok(x) => Ok(x),
        Err(x) => Err(x),
    }
}

// CHECK-LABEL: @result_nop_traits_128
#[no_mangle]
pub fn result_nop_traits_128(x: Result<i128, u128>) -> Result<i128, u128> {
    // CHECK: start:
    // CHECK-NEXT: getelementptr inbounds {{(nuw )?}}i8
    // CHECK-NEXT: store i128
    // CHECK-NEXT: store i128
    // CHECK-NEXT: ret void
    try { x? }
}

// CHECK-LABEL: @control_flow_nop_match_128
#[no_mangle]
pub fn control_flow_nop_match_128(x: ControlFlow<i128, u128>) -> ControlFlow<i128, u128> {
    // CHECK: start:
    // CHECK-NEXT: store i128
    // CHECK-NEXT: getelementptr inbounds {{(nuw )?}}i8
    // CHECK-NEXT: store i128
    // CHECK-NEXT: ret void
    match x {
        Continue(x) => Continue(x),
        Break(x) => Break(x),
    }
}

// CHECK-LABEL: @control_flow_nop_traits_128
#[no_mangle]
pub fn control_flow_nop_traits_128(x: ControlFlow<i128, u128>) -> ControlFlow<i128, u128> {
    // CHECK: start:
    // CHECK-NEXT: getelementptr inbounds {{(nuw )?}}i8
    // CHECK-NEXT: store i128
    // CHECK-NEXT: store i128
    // CHECK-NEXT: ret void
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
