// Under eager mono item collection (`-Clink-dead-code`), collecting the provided method of a trait
// implemented for (a type that normalizes to) its own trait object type used to resolve the method
// to a virtual dispatch, which has no MIR body, and crashed the collector with "virtual dispatches
// have no instance MIR". Such impls are accepted by coherence, so the program should compile.
//
// Regression test for #158411.

//@ build-pass
//@ compile-flags: -Clink-dead-code

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

// (1) Self type is a lazy type alias resolving to `dyn Trait` (the original #158411 repro).
pub trait Trait {
    fn a(&self) {}
}
pub type Alias = dyn Trait;
impl Trait for Alias {}

// (2) Self type is an associated type projection resolving to `dyn Bar`. This collects a *provided*
//     method through a projection self type (the #141119 test only exercises a required method).
pub trait Bar {
    fn b(&self) {}
}
pub trait Mirror {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type Assoc = T;
}
impl Bar for <dyn Bar as Mirror>::Assoc {}

// (3) Multiple provided methods with one overridden: `one` is a concrete `Item` and must still be
//     collected, while the inherited `two` resolves to a (skipped) virtual dispatch.
pub trait Baz {
    fn one(&self) {}
    fn two(&self) {}
}
pub type BazAlias = dyn Baz;
impl Baz for BazAlias {
    fn one(&self) {}
}

// (4) Auto-trait component in the trait object type.
pub trait Qux {
    fn q(&self) {}
}
pub type QuxAlias = dyn Qux + Send;
impl Qux for QuxAlias {}

fn main() {}
