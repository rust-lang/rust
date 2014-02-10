// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(simd)];

#[simd]
struct vec4<T>(T, T, T, T); //~ ERROR SIMD vector cannot be generic

#[simd]
struct empty; //~ ERROR SIMD vector cannot be empty

#[simd]
struct i64f64(i64, f64); //~ ERROR SIMD vector should be homogeneous

#[simd]
struct int4(int, int, int, int); //~ ERROR SIMD vector element type should be machine type

fn main() {}
