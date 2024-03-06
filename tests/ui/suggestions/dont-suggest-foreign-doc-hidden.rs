//@ aux-build:hidden-struct.rs
//@ compile-flags: --crate-type lib

extern crate hidden_struct;

#[doc(hidden)]
mod local {
    pub struct Foo;
}

pub fn test(_: Foo) {}
//~^ ERROR cannot find type `Foo` in this scope

pub fn test2(_: Bar) {}
//~^ ERROR cannot find type `Bar` in this scope
