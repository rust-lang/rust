//@ known-bug: #119381

#![feature(with_negative_coherence)]
trait Trait {}
impl<const N: u8> Trait for [(); N] {}
impl<const N: i8> Trait for [(); N] {}
