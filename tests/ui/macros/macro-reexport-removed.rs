//@ aux-build:two_macros.rs
//@ normalize-stderr: "you are using [0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?( \([^)]*\))?" -> "you are using $$RUSTC_VERSION"

#![feature(macro_reexport)] //~ ERROR feature has been removed

#[macro_reexport(macro_one)] //~ ERROR cannot find attribute `macro_reexport` in this scope
extern crate two_macros;

fn main() {}
