// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(attr_literals)]

#[path = 1usize] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1u8] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1u16] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1u32] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1u64] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1isize] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1i8] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1i16] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1i32] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1i64] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1.0f32] //~ ERROR: suffixed literals are not allowed in attributes
#[path = 1.0f64] //~ ERROR: suffixed literals are not allowed in attributes
fn main() { }
