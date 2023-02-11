#![feature(dyn_star)]
#![allow(incomplete_features)]

trait Tr {}

fn f(x: dyn* Tr) -> usize {
    x as usize
    //~^ ERROR casting `(dyn* Tr + 'static)` as `usize` is invalid
}

fn main() {}
