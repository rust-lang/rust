#![feature(fn_delegation)]

struct X;

impl X {
    fn foo(&self) {}
    fn foo2(&self) {}
}

struct Y;

impl Y {
    reuse X::{foo, foo2} { X }
}

impl Y {
    reuse X::*;
    //~^ ERROR: expected trait, found struct `X`
}

fn main() {}
