#![allow(incomplete_features)]
#![feature(const_trait_impl, try_trait_v2, const_try)]
use std::ops::{FromResidual, Try};

struct TryMe;
struct Error;

impl const FromResidual<Error> for TryMe {}
//~^ ERROR not all trait items implemented

impl const Try for TryMe {
    //~^ ERROR not all trait items implemented
    type Output = ();
    type Residual = Error;
}

const fn t() -> TryMe {
    TryMe?;
    TryMe
}

const _: () = {
    t();
};

fn main() {}
