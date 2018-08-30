// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Copy,Clone)]
struct Functions {
    a: fn(u32) -> u32,
    b: extern "C" fn(u32) -> u32,
    c: unsafe fn(u32) -> u32,
    d: unsafe extern "C" fn(u32) -> u32
}

pub fn main() {}
