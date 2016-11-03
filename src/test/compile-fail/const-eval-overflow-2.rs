// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Evaluation of constants in refutable patterns goes through
// different compiler control-flow paths.

#![allow(unused_imports)]

use std::fmt;
use std::{i8, i16, i32, i64, isize};
use std::{u8, u16, u32, u64, usize};

const NEG_128: i8 = -128;
const NEG_NEG_128: i8 = -NEG_128;
//~^ ERROR constant evaluation error
//~| attempt to negate with overflow

fn main() {
    match -128i8 {
        NEG_NEG_128 => println!("A"), //~ NOTE for pattern here
        _ => println!("B"),
    }
}
