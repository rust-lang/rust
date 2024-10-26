//@ compile-flags: -Znext-solver

// Demonstrates what's needed to make use of `?` in const contexts.

#![crate_type = "lib"]
#![feature(try_trait_v2)]
#![feature(const_trait_impl, effects)]
#![feature(const_try)]
#![allow(incomplete_features)]

use std::ops::{ControlFlow, FromResidual, Try};

struct TryMe;
struct Error;

impl const FromResidual<Error> for TryMe {
    //~^ ERROR const `impl` for trait `FromResidual` which is not marked with `#[const_trait]`
    fn from_residual(residual: Error) -> Self {
        TryMe
    }
}

impl const Try for TryMe {
    //~^ ERROR const `impl` for trait `Try` which is not marked with `#[const_trait]`
    type Output = ();
    type Residual = Error;
    fn from_output(output: Self::Output) -> Self {
        TryMe
    }
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        ControlFlow::Break(Error)
    }
}

const fn t() -> TryMe {
    TryMe?;
    //~^ ERROR `?` cannot determine the branch of `TryMe` in constant functions
    //~| ERROR `?` cannot convert from residual of `TryMe` in constant functions
    TryMe
}

const _: () = {
    t();
};
