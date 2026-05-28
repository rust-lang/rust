//@ check-fail

//@ aux-build:crateresolve2-1.rs
//@ aux-build:crateresolve2-2.rs
//@ aux-build:crateresolve2-3.rs

//@ normalize-stderr: "crateresolve2\..+/auxiliary/" -> "crateresolve2/auxiliary/"
//@ normalize-stderr: "\\\?\\" -> ""

extern crate crateresolve2;
//~^ ERROR multiple candidates for `rmeta` dependency `crateresolve2` found

fn main() {}
