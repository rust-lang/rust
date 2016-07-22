// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Evaluation of constants in array-elem count goes through different
// compiler control-flow paths.
//
// This test is checking the count in an array type.

#![allow(unused_imports)]

use std::{i8, i16, i32, i64, isize};
use std::{u8, u16, u32, u64, usize};

const A_I8_T
    : [u32; (i8::MAX as i8 + 1u8) as usize]
    //~^ ERROR constant evaluation error [E0080]
    //~| expected i8, found u8
    = [0; (i8::MAX as usize) + 1];


const A_CHAR_USIZE
    : [u32; 5u8 as char as usize]
    = [0; 5];


const A_BAD_CHAR_USIZE
    : [u32; 5i8 as char as usize]
    //~^ ERROR constant evaluation error
    //~| only `u8` can be cast as `char`, not `i8`
    = [0; 5];

fn main() {}
