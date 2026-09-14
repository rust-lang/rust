//@ aux-build:unused_extern_crate_decl.rs
//@ aux-build:unused_extern_crate_impl.rs
// Tests that dependencies that contain an EII decl without any EII impl are
// still considered unused.
#![feature(extern_item_impls)]
#![deny(unused_extern_crates)]

extern crate unused_extern_crate_decl; //~ ERROR unused extern crate
extern crate unused_extern_crate_impl;

fn main() {}
