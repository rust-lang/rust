//@ aux-build:minicore.rs
//@ compile-flags: --crate-type=lib -Znext-solver

#![feature(no_core, const_trait_impl, effects)]
//~^ WARN the feature `effects` is incomplete
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct Ty;
impl Deref for Ty {
    type Target = ();
    fn deref(&self) -> &Self::Target { &() }
}

const fn foo() {
    *Ty;
    //~^ ERROR the trait bound `Ty: ~const minicore::Deref` is not satisfied
}
