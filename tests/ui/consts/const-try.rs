//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver

// Demonstrates what's needed to make use of `?` in const contexts.

#![crate_type = "lib"]
#![feature(try_trait_v2)]
#![feature(const_trait_impl)]
#![feature(const_try)]

use std::ops::{ControlFlow, FromResidual, Branch, FromOutput};

struct TryMe;
struct Error;

impl const FromResidual<Error> for TryMe {
    fn from_residual(residual: Error) -> Self {
        TryMe
    }
}

impl const Branch for TryMe {
    type Output = ();
    type Residual = Error;
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        ControlFlow::Break(Error)
    }
}

impl const FromOutput for TryMe {
    fn from_output(output: Self::Output) -> Self {
        TryMe
    }
}

const fn t() -> TryMe {
    TryMe?;
    TryMe
}

const _: () = {
    t();
};
