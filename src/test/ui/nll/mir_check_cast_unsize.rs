// compile-flags: -Z borrowck=mir

#![allow(dead_code)]

use std::fmt::Debug;

fn bar<'a>(x: &'a u32) -> &'static dyn Debug {
    //~^ ERROR unsatisfied lifetime constraints
    x
    //~^ WARNING not reporting region error due to nll
}

fn main() {}
