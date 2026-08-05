//@ run-pass
#![allow(unused_imports)]
use std::fmt;

const A_I8_T
    : [u32; (i8::MAX as i8 - 1i8) as usize]
    = [0; (i8::MAX as usize) - 1];

fn main() {
    foo(&A_I8_T[..]);
}

fn foo<T:fmt::Debug>(x: T) {
    println!("{:?}", x);
}
