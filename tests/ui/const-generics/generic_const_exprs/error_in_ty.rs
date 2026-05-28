//@ compile-flags: -Znext-solver=coherence

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub struct A<const z: [usize; x]> {}
//~^ ERROR: cannot find value `x` in this scope
//~| ERROR: `[usize; x]` is forbidden as the type of a const generic parameter

impl A<2> {
    //~^ ERROR: mismatched types
    pub const fn B() {}
    //~^ ERROR: duplicate definitions
}

impl A<2> {
    //~^ ERROR: mismatched types
    pub const fn B() {}
}

fn main() {}
