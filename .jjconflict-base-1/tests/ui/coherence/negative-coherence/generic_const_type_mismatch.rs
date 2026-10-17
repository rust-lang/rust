//! This test used to ICE (#119381), because relating the `u8` and `i8` generic
//! const with the array length of the `Self` type was succeeding under the
//! assumption that an error had already been reported.

#![feature(with_negative_coherence)]
trait Trait {}
impl<const N: u8> Trait for [(); N] {}
impl<const N: i8> Trait for [(); N] {}
//~^ ERROR: conflicting implementations of trait `Trait`

fn main() {}
