#![feature(dyn_star)]
#![allow(incomplete_features)]

trait Tr {}

fn f(x: dyn* Tr) -> usize {
    x as usize
    //~^ ERROR non-primitive cast: `(dyn* Tr + 'static)` as `usize`
}

fn main() {}
