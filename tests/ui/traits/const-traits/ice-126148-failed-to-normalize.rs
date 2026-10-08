#![allow(incomplete_features)]
#![feature(const_trait_impl, try_trait_v2, const_try)]
use std::ops::{TryFromBreak, Try};

struct TryMe;
struct Error;

const impl TryFromBreak<Error> for TryMe {}
//~^ ERROR not all trait items implemented

const impl Try for TryMe {
    //~^ ERROR not all trait items implemented
    type Output = ();
    type Break = Error;
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
