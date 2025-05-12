// test for ICE #119275 "no entry found for key" in predicates_of.rs

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn bug<const N: Nat>(&self)
//~^ ERROR `self` parameter is only allowed in associated functions
//~^^ ERROR cannot find type `Nat` in this scope
where
    for<const N: usize = 3, T = u32> [(); COT::BYTES]:,
    //~^ ERROR only lifetime parameters can be used in this context
    //~| ERROR defaults for generic parameters are not allowed in `for<...>` binders
    //~| ERROR defaults for generic parameters are not allowed in `for<...>` binders
    //~| ERROR failed to resolve: use of undeclared type `COT`
    //~| ERROR  the name `N` is already used for a generic parameter in this item's generic parameters
{
}

pub fn main() {}
