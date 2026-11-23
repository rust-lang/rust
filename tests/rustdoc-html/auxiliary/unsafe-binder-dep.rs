//@ compile-flags: -Znext-solver

#![feature(unsafe_binders)]
#![allow(incomplete_features)]

pub fn woof() -> unsafe<'a> &'a str { todo!() }
