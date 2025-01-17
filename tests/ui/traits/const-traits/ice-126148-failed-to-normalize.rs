#![allow(incomplete_features)]
#![feature(const_trait_impl, try_trait_v2, const_try)]
use std::ops::{FromResidual, Try};

struct TryMe;
struct Error;

impl const FromResidual<Error> for TryMe {}
//~^ ERROR const `impl` for trait `FromResidual` which is not marked with `#[const_trait]`
//~| ERROR not all trait items implemented

impl const Try for TryMe {
    //~^ ERROR const `impl` for trait `Try` which is not marked with `#[const_trait]`
    //~| ERROR not all trait items implemented
    type Output = ();
    type Residual = Error;
}

const fn t() -> TryMe {
    TryMe?;
    //~^ ERROR `?` is not allowed on
    //~| ERROR `?` is not allowed on
    TryMe
}

const _: () = {
    t();
};

fn main() {}
