//@ known-bug: rust-lang/rust#124857
//@ compile-flags: -Znext-solver=coherence

#![feature(effects)]

#[const_trait]
trait Foo {}

impl const Foo for i32 {}

impl<T> const Foo for T where T: ~const Foo {}
