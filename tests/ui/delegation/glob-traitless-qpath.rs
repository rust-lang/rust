#![feature(fn_delegation)]

struct S;

impl S {
    reuse <u8>::*; //~ ERROR qualified path without a trait in glob delegation
    reuse <()>::*; //~ ERROR qualified path without a trait in glob delegation
}

fn main() {}
