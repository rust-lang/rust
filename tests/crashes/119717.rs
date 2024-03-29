//@ known-bug: #119717
#![feature(const_trait_impl, effects)]

use std::ops::{FromResidual, Try};

impl const FromResidual for T {
    fn from_residual(t: T) -> _ {
        t
    }
}
