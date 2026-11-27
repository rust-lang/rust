//! `with_negative_coherence` in the trait's crate (core) must not apply to this crate.

use std::ops::DerefMut;

struct M;

impl<T: DerefMut> From<T> for M {
    fn from(_: T) -> M {
        M
    }
}

impl<'a, T> From<&'a T> for M {
    //~^ ERROR conflicting implementations of trait `From<&_>` for type `M`
    fn from(_: &'a T) -> M {
        M
    }
}

fn main() {}
