//@ aux-build:crateresolve1-1.rs
//@ aux-build:crateresolve1-2.rs
//@ aux-build:crateresolve1-3.rs

//@ normalize-stderr-test: "\.nll/" -> "/"
//@ normalize-stderr-test: "\\\?\\" -> ""
//@ normalize-stderr-test: "(lib)?crateresolve1-([123])\.[a-z]+" -> "libcrateresolve1-$2.somelib"

// NOTE: This test is duplicated from `tests/ui/crate-loading/crateresolve1.rs`.

extern crate crateresolve1;
//~^ ERROR multiple candidates for `rlib` dependency `crateresolve1` found

fn main() {}
