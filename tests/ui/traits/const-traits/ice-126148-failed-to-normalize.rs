#![allow(incomplete_features)]
#![feature(const_trait_impl, try_trait_v2, try_trait_v2_residual, const_try, const_try_residual)]
use std::ops::{FromResidual, Residual, Try};

struct TryMe;
struct Error;

impl const FromResidual<Error> for TryMe {}
//~^ ERROR not all trait items implemented

impl const Try for TryMe {
    //~^ ERROR not all trait items implemented
    type Output = ();
    type Residual = Error;
}

impl const Residual<()> for Error {
    type TryType = TryMe;
}

const fn t() -> TryMe {
    TryMe?;
    TryMe
}

const _: () = {
    t();
};

fn main() {}
