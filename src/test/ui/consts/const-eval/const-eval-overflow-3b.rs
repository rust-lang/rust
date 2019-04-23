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
//~| ERROR cannot add `u8` to `i8`

fn main() {
    foo(&A_I8_I[..]);
}

fn foo<T:fmt::Debug>(x: T) {
    println!("{:?}", x);
}
