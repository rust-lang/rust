#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {}

struct S;
impl S {
    reuse Trait::*; //~ ERROR empty glob delegation is not supported
}

fn main() {}
