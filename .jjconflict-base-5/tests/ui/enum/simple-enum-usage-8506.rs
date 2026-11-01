// https://github.com/rust-lang/rust/issues/8506
//@ run-pass
#![allow(non_upper_case_globals)]

#![allow(dead_code)]

enum Either {
    One,
    Other(String,String)
}

static one : Either = Either::One;

pub fn main () { }
