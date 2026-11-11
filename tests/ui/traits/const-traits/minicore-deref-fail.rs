//@ aux-build:minicore.rs
//@ compile-flags: --crate-type=lib -Znext-solver -Cpanic=abort

#![feature(no_core, const_trait_impl)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct Ty;
impl Deref for Ty {
    type Target = ();
    fn deref(&self) -> &Self::Target {
        &()
    }
}

const fn foo() {
    *Ty;
    //~^ ERROR the trait bound `Ty: [const] minicore::Deref` is not satisfied
}
