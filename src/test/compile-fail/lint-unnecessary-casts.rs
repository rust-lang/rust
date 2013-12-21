// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[forbid(unnecessary_typecast)];

fn foo_i32(_: i32) {}

fn foo_u64(a: u64) {
    let b: i32 = a as i32;
    foo_i32(b as i32); //~ ERROR: unnecessary type cast
}

fn main() {
    let x: u64 = 1;
    let y: u64 = x as u64; //~ ERROR: unnecessary type cast
    foo_u64(y as u64); //~ ERROR: unnecessary type cast
}
