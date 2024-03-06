//@ check-pass

#![feature(with_negative_coherence)]

use std::ops::DerefMut;

trait Foo {}
impl<T: DerefMut> Foo for T {}
impl<U> Foo for &U {}

fn main() {}
