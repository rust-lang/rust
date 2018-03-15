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

#[no_mangle]
pub struct F32(f32);

// CHECK: define float @add_newtype_f32(float, float)
#[inline(never)]
#[no_mangle]
pub fn add_newtype_f32(a: F32, b: F32) -> F32 {
    F32(a.0 + b.0)
}

#[no_mangle]
pub struct F64(f64);

// CHECK: define double @add_newtype_f64(double, double)
#[inline(never)]
#[no_mangle]
pub fn add_newtype_f64(a: F64, b: F64) -> F64 {
    F64(a.0 + b.0)
}
