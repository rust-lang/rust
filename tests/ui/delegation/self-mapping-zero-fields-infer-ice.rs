#![feature(fn_delegation)]

trait Trait {
    fn add(self) -> Self;
}

struct S;
impl S {
    reuse Trait::*;
    //~^ ERROR: the delegation parent does not have a field for `Self` mapping
}

fn main() {}
