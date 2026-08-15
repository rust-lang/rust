//@ known-bug: #148629
#![feature(with_negative_coherence)]
trait Foo {
    type AssociatedType;
}

impl<const N: usize> Foo for [(); N] {}

pub struct Happy;

impl Foo for Happy {}

impl<const N: usize> Foo for Happy where <[(); N] as Foo>::AssociatedType: Clone {}
