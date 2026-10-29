//@ check-fail

#![feature(specialization)]
#![allow(incomplete_features)]

// `default impl` still participates in coherence. However, we shouldn't get an overflow here.
// Regresion test for #77026.

pub enum Either<L, R> {
    Left(L),
    Right(R),
}

default impl<L, R> From<L> for Either<L, R> {
    fn from(l: L) -> Self {
        Either::Left(l)
    }
}

impl<L, R> From<R> for Either<L, R> {
    //~^ ERROR conflicting implementations of trait `From<_>` for type `Either<_, _>`
    fn from(r: R) -> Self {
        Either::Right(r)
    }
}

fn main() {}
