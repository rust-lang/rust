// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::{fadd_fast, fsub_fast, fmul_fast, fdiv_fast, frem_fast};

// CHECK-LABEL: @add
#[no_mangle]
pub fn add(x: f32, y: f32) -> f32 {
// CHECK: fadd float
// CHECK-NOT fast
    x + y
}

// CHECK-LABEL: @addition
#[no_mangle]
pub fn addition(x: f32, y: f32) -> f32 {
// CHECK: fadd fast float
    unsafe {
        fadd_fast(x, y)
    }
}

// CHECK-LABEL: @subtraction
#[no_mangle]
pub fn subtraction(x: f32, y: f32) -> f32 {
// CHECK: fsub fast float
    unsafe {
        fsub_fast(x, y)
    }
}

// CHECK-LABEL: @multiplication
#[no_mangle]
pub fn multiplication(x: f32, y: f32) -> f32 {
// CHECK: fmul fast float
    unsafe {
        fmul_fast(x, y)
    }
}

// CHECK-LABEL: @division
#[no_mangle]
pub fn division(x: f32, y: f32) -> f32 {
// CHECK: fdiv fast float
    unsafe {
        fdiv_fast(x, y)
    }
}
