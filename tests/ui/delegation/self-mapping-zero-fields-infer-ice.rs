#![feature(fn_delegation)]

trait Trait {
    fn add(self) -> Self;
}

struct S;
impl S {
    reuse Trait::*;
    //~^ ERROR: the `S` must contain one field for wrapping the return value in delegation
}

fn main() {}
