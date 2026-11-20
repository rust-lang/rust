// test for ICE "no entry found for key" in generics_of.rs #113017

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub fn foo()
where
    for<const N: usize = { || {}; 1 }> ():,
    //~^ ERROR only lifetime parameters can be used in this context
    //~^^  ERROR defaults for generic parameters are not allowed in `for<...>` binders
{}

pub fn main() {}
