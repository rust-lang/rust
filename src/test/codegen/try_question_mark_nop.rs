// min-llvm-version: 15.0
// compile-flags: -O -Z merge-functions=disabled --edition=2021
// only-x86_64

#![crate_type = "lib"]
#![feature(try_blocks)]

// These are now NOPs in LLVM 15, presumably thanks to nikic's change mentioned in
// <https://github.com/rust-lang/rust/issues/85133#issuecomment-1072168354>.
// Unfortunately, as of 2022-08-17 they're not yet nops for `u64`s nor `Option`.

use std::ops::ControlFlow::{self, Continue, Break};

// CHECK-LABEL: @result_nop_match_32
#[no_mangle]
pub fn result_nop_match_32(x: Result<i32, u32>) -> Result<i32, u32> {
    // CHECK: start
    // CHECK-NEXT: ret i64 %0
    match x {
        Ok(x) => Ok(x),
        Err(x) => Err(x),
    }
}

// CHECK-LABEL: @result_nop_traits_32
#[no_mangle]
pub fn result_nop_traits_32(x: Result<i32, u32>) -> Result<i32, u32> {
    // CHECK: start
    // CHECK-NEXT: ret i64 %0
    try {
        x?
    }
}

// CHECK-LABEL: @control_flow_nop_match_32
#[no_mangle]
pub fn control_flow_nop_match_32(x: ControlFlow<i32, u32>) -> ControlFlow<i32, u32> {
    // CHECK: start
    // CHECK-NEXT: ret i64 %0
    match x {
        Continue(x) => Continue(x),
        Break(x) => Break(x),
    }
}

// CHECK-LABEL: @control_flow_nop_traits_32
#[no_mangle]
pub fn control_flow_nop_traits_32(x: ControlFlow<i32, u32>) -> ControlFlow<i32, u32> {
    // CHECK: start
    // CHECK-NEXT: ret i64 %0
    try {
        x?
    }
}
