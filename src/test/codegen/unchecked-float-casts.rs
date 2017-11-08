// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

// This file tests that we don't generate any code for saturation if
// -Z saturating-float-casts is not enabled.

#![crate_type = "lib"]
#![feature(i128_type)]

// CHECK-LABEL: @f32_to_u32
#[no_mangle]
pub fn f32_to_u32(x: f32) -> u32 {
    // CHECK: fptoui
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    x as u32
}

// CHECK-LABEL: @f32_to_i32
#[no_mangle]
pub fn f32_to_i32(x: f32) -> i32 {
    // CHECK: fptosi
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    x as i32
}

#[no_mangle]
pub fn f64_to_u8(x: f32) -> u16 {
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    x as u16
}

// CHECK-LABEL: @i32_to_f64
#[no_mangle]
pub fn i32_to_f64(x: i32) -> f64 {
    // CHECK: sitofp
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    x as f64
}

// CHECK-LABEL: @u128_to_f32
#[no_mangle]
pub fn u128_to_f32(x: u128) -> f32 {
    // CHECK: uitofp
    // CHECK-NOT: fcmp
    // CHECK-NOT: icmp
    // CHECK-NOT: select
    x as f32
}
