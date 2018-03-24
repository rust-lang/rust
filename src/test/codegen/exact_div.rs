// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
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

use std::intrinsics::exact_div;

// CHECK-LABEL: @exact_sdiv
#[no_mangle]
pub unsafe fn exact_sdiv(x: i32, y: i32) -> i32 {
// CHECK: sdiv exact
    exact_div(x, y)
}

// CHECK-LABEL: @exact_udiv
#[no_mangle]
pub unsafe fn exact_udiv(x: u32, y: u32) -> u32 {
// CHECK: udiv exact
    exact_div(x, y)
}
