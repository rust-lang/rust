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
// This test is checking the count in an array expression.
//
// This is a variation of another such test, but in this case the
// types for the left- and right-hand sides of the addition do not
// match (as well as overflow).

#![allow(unused_imports)]

use std::fmt;
use std::{i8, i16, i32, i64, isize};
use std::{u8, u16, u32, u64, usize};

const A_I8_I
    : [u32; (i8::MAX as usize) + 1]
    = [0; (i8::MAX + 1u8) as usize];
//~^ ERROR mismatched types
//~| ERROR mismatched types
//~| ERROR expected constant integer for repeat count, but attempted to add with overflow
//~| ERROR the trait `core::ops::Add<u8>` is not implemented for the type `i8`
//~| ERROR the trait `core::ops::Add<u8>` is not implemented for the type `i8`

fn main() {
    foo(&A_I8_I[..]);
}

fn foo<T:fmt::Debug>(x: T) {
    println!("{:?}", x);
}

