// Evaluation of constants in array-elem count goes through different
// compiler control-flow paths.
//
// This test is checking the count in an array expression.







#![allow(unused_imports)]

use std::fmt;

const A_I8_I
    : [u32; (i8::MAX as usize) + 1]
    = [0; (i8::MAX + 1) as usize];
//~^ ERROR overflow

fn main() {
    foo(&A_I8_I[..]);
}

fn foo<T:fmt::Debug>(x: T) {
    println!("{:?}", x);
}
