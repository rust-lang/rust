//@ aux-build:drop-shim-relates-opaque-aux.rs
//@ compile-flags: -Zvalidate-mir --crate-type=lib
//@ build-pass

extern crate drop_shim_relates_opaque_aux;

pub fn drop_foo(_: drop_shim_relates_opaque_aux::Foo) {}
pub fn drop_bar(_: drop_shim_relates_opaque_aux::Bar) {}

fn main() {}
