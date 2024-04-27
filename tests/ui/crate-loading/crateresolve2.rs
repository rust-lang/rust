//@ check-fail

//@ aux-build:crateresolve2-1.rs
//@ aux-build:crateresolve2-2.rs
//@ aux-build:crateresolve2-3.rs

//@ normalize-stderr-test: "\.nll/" -> "/"
//@ normalize-stderr-test: "\\\?\\" -> ""

extern crate crateresolve2;
//~^ ERROR multiple candidates for `rmeta` dependency `crateresolve2` found

fn main() {}
