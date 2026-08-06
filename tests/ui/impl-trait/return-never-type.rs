//@ edition:2024

#![feature(never_type)]

use std::ops::Add;

fn foo() -> impl Add<u32> {
    //~^ ERROR cannot add `u32` to `!`
    //~| HELP the trait `Add<u32>` is not implemented for `!`
    unimplemented!()
    //~^ HELP `!` can be coerced to any type; consider casting it to a concrete type that implements the trait
}

fn main() {}
