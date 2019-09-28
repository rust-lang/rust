#![feature(re_rebalance_coherence)]

// compile-flags:--crate-name=test
// aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;

impl<T> Remote2<Box<T>, Local> for u32 {
    //~^ ERROR type parameter `T` must be used as the type parameter for some local type
}

impl<'a, T> Remote2<&'a T, Local> for u32 {
    //~^ ERROR type parameter `T` must be used as the type parameter for some local type
}

fn main() {}
