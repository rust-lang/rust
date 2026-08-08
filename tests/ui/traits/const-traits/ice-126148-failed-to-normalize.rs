#![allow(incomplete_features)]
#![feature(const_trait_impl, const_try_residual, try_trait_v2, try_trait_v2_residual, const_try)]
use std::ops::{Branch, FromOutput, FromResidual, Residual};

struct TryMe;
struct Error;

const impl FromResidual<Error> for TryMe {}
//~^ ERROR not all trait items implemented

const impl Branch for TryMe {
    //~^ ERROR not all trait items implemented
    type Output = ();
    type Residual = Error;
}
const impl FromOutput<()> for TryMe {
    //~^ ERROR not all trait items implemented
}

const impl Residual<()> for Error {
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
