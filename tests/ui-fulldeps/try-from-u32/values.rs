#![feature(assert_matches)]
#![feature(rustc_private)]
//@ edition: 2021
//@ run-pass

// Checks the values accepted by the `TryFrom<u32>` impl produced by `#[derive(TryFromU32)]`.

extern crate rustc_macros;

use core::assert_matches::assert_matches;
use rustc_macros::TryFromU32;

#[derive(TryFromU32, Debug, PartialEq)]
#[repr(u32)]
enum Repr {
    Zero,
    One(),
    Seven = 7,
}

#[derive(TryFromU32, Debug)]
enum NoRepr {
    Zero,
    One,
}

fn main() {
    assert_eq!(Repr::try_from(0u32), Ok(Repr::Zero));
    assert_eq!(Repr::try_from(1u32), Ok(Repr::One()));
    assert_eq!(Repr::try_from(2u32), Err(2));
    assert_eq!(Repr::try_from(7u32), Ok(Repr::Seven));

    assert_matches!(NoRepr::try_from(0u32), Ok(NoRepr::Zero));
    assert_matches!(NoRepr::try_from(1u32), Ok(NoRepr::One));
    assert_matches!(NoRepr::try_from(2u32), Err(2));
}
