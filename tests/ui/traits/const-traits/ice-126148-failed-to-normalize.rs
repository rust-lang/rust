#![allow(incomplete_features)]
#![feature(const_trait_impl, try_trait_v2, const_try, const_try_residual)]
use std::ops::{FromResidual, Residual, Try};

struct TryMe;
struct Error;

const impl FromResidual<Error> for TryMe {}
//~^ ERROR not all trait items implemented

const impl Try for TryMe {
    //~^ ERROR not all trait items implemented
    type Output = ();
    type Residual = Error;
}

const impl TryAs<()> for Error {
    type Try = TryMe;
}

const fn t() -> TryMe {
    TryMe?;
    TryMe
}

const _: () = {
    t();
};

fn main() {}
