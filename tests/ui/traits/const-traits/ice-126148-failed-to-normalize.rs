#![allow(incomplete_features)]
#![feature(
    const_trait_impl,
    const_try_residual,
    try_trait_v2,try_trait_v2_residual,
    const_try
)]
use std::ops::{Branch, FromResidual, Residual, FromOutput};

struct TryMe;
struct Error;

impl const FromResidual<Error> for TryMe {}
//~^ ERROR not all trait items implemented

impl const Branch for TryMe {
    //~^ ERROR not all trait items implemented
    type Output = ();
    type Residual = Error;
}
impl const FromOutput<()> for TryMe {
    //~^ ERROR not all trait items implemented
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
