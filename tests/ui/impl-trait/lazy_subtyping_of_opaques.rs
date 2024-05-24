//! This test checks that we allow subtyping predicates that contain opaque types.
//! No hidden types are being constrained in the subtyping predicate, but type and
//! lifetime variables get subtyped in the generic parameter list of the opaque.

use std::iter;

mod either {
    pub enum Either<L, R> {
        Left(L),
        Right(R),
    }

    impl<L: Iterator, R: Iterator<Item = L::Item>> Iterator for Either<L, R> {
        type Item = L::Item;
        fn next(&mut self) -> Option<Self::Item> {
            todo!()
        }
    }
    pub use self::Either::{Left, Right};
}

pub enum BabeConsensusLogRef<'a> {
    NextEpochData(BabeNextEpochRef<'a>),
    NextConfigData,
}

impl<'a> BabeConsensusLogRef<'a> {
    pub fn scale_encoding(
        &self,
    ) -> impl Iterator<Item = impl AsRef<[u8]> + Clone + 'a> + Clone + 'a {
        //~^ ERROR is not satisfied
        //~| ERROR is not satisfied
        //~| ERROR is not satisfied
        match self {
            BabeConsensusLogRef::NextEpochData(digest) => either::Left(either::Left(
                digest.scale_encoding().map(either::Left).map(either::Left),
            )),
            BabeConsensusLogRef::NextConfigData => either::Right(
                // The Opaque type from ``scale_encoding` gets used opaquely here, while the `R`
                // generic parameter of `Either` contains type variables that get subtyped and the
                // opaque type contains lifetime variables that get subtyped.
                iter::once(either::Right(either::Left([1])))
                    .chain(std::iter::once([1]).map(either::Right).map(either::Right)),
            ),
        }
    }
}

pub struct BabeNextEpochRef<'a>(&'a ());

impl<'a> BabeNextEpochRef<'a> {
    pub fn scale_encoding(
        &self,
    ) -> impl Iterator<Item = impl AsRef<[u8]> + Clone + 'a> + Clone + 'a {
        std::iter::once([1])
    }
}

fn main() {}
