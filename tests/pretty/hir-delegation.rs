//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-delegation.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]

fn b<C>(e: C) {}

trait G {
    reuse b {}
}

mod m {
    pub fn add(a: u32, b: u32) -> u32 { a + b }
}

reuse m::add;

fn main() {
    _ = add(1, 2);
}
