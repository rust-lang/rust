// compile-flags: --crate-type=lib
#![feature(impl_restriction)]

pub impl trait Foo {} //~ ERROR incorrect impl restriction
