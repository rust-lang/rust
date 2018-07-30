// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test several functions can be used for constants
// 1. Vec::new()
// 2. String::new()

#![feature(const_vec_new)]
#![feature(const_string_new)]

const MY_VEC: Vec<usize> = Vec::new();

const MY_STRING: String = String::new();

pub fn main() {}
