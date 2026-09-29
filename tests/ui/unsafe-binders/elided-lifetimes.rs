//@ compile-flags: -Znext-solver

// Elided and `'_` lifetimes can't appear inside an unsafe binder.

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

type Elided = unsafe<> &u8;
//~^ ERROR `&` without an explicit lifetime name cannot be used here

type Underscore = unsafe<> &'_ u8;
//~^ ERROR `'_` cannot be used here

fn arg(_: unsafe<> &u8) {}
//~^ ERROR `&` without an explicit lifetime name cannot be used here

fn main() {}
