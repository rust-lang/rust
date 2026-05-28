//@ compile-flags: --crate-type=lib

#![no_implicit_prelude]

#[deprecated(suggestion = "foo")] //~ ERROR suggestions on deprecated items are unstable
struct Foo {}
