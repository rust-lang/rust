// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::*;

// CHECK-LABEL: @nowrap_add_signed
#[no_mangle]
pub unsafe fn nowrap_add_signed(a: i32, b: i32) -> i32 {
    // CHECK: add nsw
    nowrap_add(a, b)
}

// CHECK-LABEL: @nowrap_add_unsigned
#[no_mangle]
pub unsafe fn nowrap_add_unsigned(a: u32, b: u32) -> u32 {
    // CHECK: add nuw
    nowrap_add(a, b)
}

// CHECK-LABEL: @nowrap_sub_signed
#[no_mangle]
pub unsafe fn nowrap_sub_signed(a: i32, b: i32) -> i32 {
    // CHECK: sub nsw
    nowrap_sub(a, b)
}

// CHECK-LABEL: @nowrap_sub_unsigned
#[no_mangle]
pub unsafe fn nowrap_sub_unsigned(a: u32, b: u32) -> u32 {
    // CHECK: sub nuw
    nowrap_sub(a, b)
}

// CHECK-LABEL: @nowrap_mul_signed
#[no_mangle]
pub unsafe fn nowrap_mul_signed(a: i32, b: i32) -> i32 {
    // CHECK: mul nsw
    nowrap_mul(a, b)
}

// CHECK-LABEL: @nowrap_mul_unsigned
#[no_mangle]
pub unsafe fn nowrap_mul_unsigned(a: u32, b: u32) -> u32 {
    // CHECK: mul nuw
    nowrap_mul(a, b)
}

// CHECK-LABEL: @nowrap_neg_signed
#[no_mangle]
pub unsafe fn nowrap_neg_signed(a: i32) -> i32 {
    // CHECK: sub nsw i32 0,
    nowrap_neg(a)
}

// CHECK-LABEL: @nowrap_neg_unsigned
#[no_mangle]
pub unsafe fn nowrap_neg_unsigned(a: u32) -> u32 {
    // CHECK: ret i32 0
    nowrap_neg(a)
}
