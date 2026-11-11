//! Regression test for https://github.com/rust-lang/rust/issues/122214.
//!
//! The reported ICE only happened under incremental compilation, where the const inference
//! variable reached `HashStable`, but this used to crash without it as well, so check both.

//@ revisions: normal incr
//@[incr] incremental

#![feature(impl_trait_in_assoc_type, const_precise_live_drops)]

trait Trait {
    type Opaque1;
}

impl<const B: Word> Trait for &'a () {
    //~^ ERROR use of undeclared lifetime name `'a`
    //~| ERROR cannot find type `Word` in this scope
    type Opaque1 = impl Sized;

    fn constrain(self) -> (Self::Opaque1,) {}
    //~^ ERROR method `constrain` is not a member of trait `Trait`
}

fn main() {}
