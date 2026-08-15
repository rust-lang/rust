//@ check-pass

//! Tests that it's possible to define an associated type in a trait
//! using an associated type from type parameter bound trait in a blanket implementation.
//!
//! # Context
//! Original issue: https://github.com/rust-lang/rust/issues/19479

trait Base {
    fn dummy(&self) { }
}
trait AssocA {
    type X: Base;
    fn dummy(&self) { }
}
trait AssocB {
    type Y: Base;
    fn dummy(&self) { }
}
impl<T: AssocA> AssocB for T {
    type Y = <T as AssocA>::X;
}

fn main() {}
