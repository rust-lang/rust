// Evaluation of constants in array-elem count goes through different
// compiler control-flow paths.
//
// This test is checking the count in an array type.

#![allow(unused_imports)]

use std::fmt;
use std::{i8, i16, i32, i64, isize};
use std::{u8, u16, u32, u64, usize};

const A_I8_T
    : [u32; (i8::MAX as i8 + 1i8) as usize]
    //~^ ERROR evaluation of constant value failed
    = [0; (i8::MAX as usize) + 1];

fn main() {
    foo(&A_I8_T[..]);
}

fn foo<T:fmt::Debug>(x: T) {
    println!("{:?}", x);
}
