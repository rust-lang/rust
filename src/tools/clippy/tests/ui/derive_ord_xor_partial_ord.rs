#![warn(clippy::derive_ord_xor_partial_ord)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::incorrect_partial_ord_impl_on_ord_type)]

use std::cmp::Ordering;

#[derive(PartialOrd, Ord, PartialEq, Eq)]
struct DeriveBoth;

impl PartialEq<u64> for DeriveBoth {
    fn eq(&self, _: &u64) -> bool {
        true
    }
}

impl PartialOrd<u64> for DeriveBoth {
    fn partial_cmp(&self, _: &u64) -> Option<Ordering> {
        Some(Ordering::Equal)
    }
}

#[derive(Ord, PartialEq, Eq)]
struct DeriveOrd;

impl PartialOrd for DeriveOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(other.cmp(self))
    }
}

#[derive(Ord, PartialEq, Eq)]
struct DeriveOrdWithExplicitTypeVariable;

impl PartialOrd<DeriveOrdWithExplicitTypeVariable> for DeriveOrdWithExplicitTypeVariable {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(other.cmp(self))
    }
}

#[derive(PartialOrd, PartialEq, Eq)]
struct DerivePartialOrd;

impl std::cmp::Ord for DerivePartialOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        Ordering::Less
    }
}

#[derive(PartialOrd, PartialEq, Eq)]
struct ImplUserOrd;

trait Ord {}

// We don't want to lint on user-defined traits called `Ord`
impl Ord for ImplUserOrd {}

mod use_ord {
    use std::cmp::{Ord, Ordering};

    #[derive(PartialOrd, PartialEq, Eq)]
    struct DerivePartialOrdInUseOrd;

    impl Ord for DerivePartialOrdInUseOrd {
        fn cmp(&self, other: &Self) -> Ordering {
            Ordering::Less
        }
    }
}

fn main() {}
