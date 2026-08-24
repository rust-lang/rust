#![feature(fn_delegation)]

struct X;

impl X {
    fn foo(&self) {}
    fn foo2(&self) {}
}

struct Y;

impl Y {
    reuse X::{foo, foo2} { X }
    //~^ ERROR: cannot find function `foo` in `X`
    //~| ERROR: cannot find function `foo2` in `X`
}

impl Y {
    reuse X::*;
    //~^ ERROR: expected trait, found struct `X`
}

fn main() {}
