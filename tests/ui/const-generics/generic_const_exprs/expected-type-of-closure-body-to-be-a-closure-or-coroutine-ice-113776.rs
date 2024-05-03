// issue: rust-lang/rust#113776
// ice: expected type of closure body to be a closure or coroutine
//@ edition: 2021
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use core::ops::SubAssign;

fn f<T>(
    data: &[(); {
         let f: F = async { 1 };
         //~^ ERROR cannot find type `F` in this scope

         1
     }],
) -> impl Iterator<Item = SubAssign> {
//~^ ERROR the type parameter `Rhs` must be explicitly specified
//~| ERROR `()` is not an iterator
//~| ERROR trait objects must include the `dyn` keyword
//~| ERROR the type parameter `Rhs` must be explicitly specified [E0393]
}

pub fn main() {}
