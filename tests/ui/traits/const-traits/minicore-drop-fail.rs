//@ aux-build:minicore.rs
//@ compile-flags: --crate-type=lib -Znext-solver

#![feature(no_core, const_trait_impl, const_destruct)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct Contains<T>(T);

struct NotDropImpl;
impl Drop for NotDropImpl {
    fn drop(&mut self) {}
}

#[const_trait] trait Foo {}
impl Foo for () {}

struct Conditional<T: Foo>(T);
impl<T> const Drop for Conditional<T> where T: [const] Foo {
    fn drop(&mut self) {}
}

const fn test() {
    let _ = NotDropImpl;
    //~^ ERROR destructor of `NotDropImpl` cannot be evaluated at compile-time
    let _ = Contains(NotDropImpl);
    //~^ ERROR destructor of `Contains<NotDropImpl>` cannot be evaluated at compile-time
    let _ = Conditional(());
    //~^ ERROR destructor of `Conditional<()>` cannot be evaluated at compile-time
}

const fn drop_arbitrary<T>(_: T) {
    //~^ ERROR destructor of `T` cannot be evaluated at compile-time
}
