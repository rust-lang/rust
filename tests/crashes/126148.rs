//@ known-bug: rust-lang/rust#126148

#![feature(effects)]
use std::ops::{FromResidual, Try};

struct TryMe;
struct Error;

impl const FromResidual<Error> for TryMe {}

impl const Try for TryMe {
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
