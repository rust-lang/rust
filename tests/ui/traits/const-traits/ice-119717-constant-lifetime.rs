#![allow(incomplete_features)]
#![feature(const_trait_impl, try_trait_v2)]

use std::ops::FromResidual;

impl<T> const FromResidual for T {
    //~^ ERROR const `impl` for trait `FromResidual` which is not `const`
    //~| ERROR type parameter `T` must be used as the type parameter for some local type
    fn from_residual(t: T) -> _ {
        //~^ ERROR the placeholder `_` is not allowed
        t
    }
}

fn main() {}
