//@ aux-build:minicore.rs
//@ compile-flags: --crate-type=lib -Znext-solver

#![feature(no_core, const_trait_impl)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

const fn call_indirect<T: [const] Fn()>(t: &T) { t() }

#[const_trait]
trait Foo {}
impl Foo for () {}
const fn foo<T: [const] Foo>() {}

const fn test() {
    call_indirect(&foo::<()>);
    //~^ ERROR the trait bound `(): [const] Foo` is not satisfied
}
