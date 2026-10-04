#![allow(incomplete_features)]
#![feature(const_trait_impl, const_try, try_trait_v2)]

use std::ops::FromResidual;

const impl<T> FromResidual for T {
    //~^ ERROR type parameter `T` must be used as an argument to some local type
    fn from_residual(t: T) -> _ {
        //~^ ERROR the placeholder `_` is not allowed
        t
    }
}

fn main() {}
