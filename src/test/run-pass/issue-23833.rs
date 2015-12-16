// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::{i8, i16, i32, i64, isize};
use std::{u8, u16, u32, u64, usize};

const A_I8_T
    : [u32; (i8::MAX as i8 - 1i8) as usize]
    = [0; (i8::MAX as usize) - 1];

fn main() {
    foo(&A_I8_T[..]);
}

fn foo<T:fmt::Debug>(x: T) {
    println!("{:?}", x);
}
