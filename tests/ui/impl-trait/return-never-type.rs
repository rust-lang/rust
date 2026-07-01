//@ edition:2024

#![feature(never_type)]

use std::ops::Add;

fn foo() -> impl Add<u32> {
    //~^ ERROR cannot add `u32` to `!`
    //~| HELP the trait `Add<u32>` is not implemented for `!`
    //~| HELP `!` can be coerced to any type; consider casting to a concrete type that implements the trait, e.g. `unimplemented!() as SomeType`
    unimplemented!()
}

fn main() {}
