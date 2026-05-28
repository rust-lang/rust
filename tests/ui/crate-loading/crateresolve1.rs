//@ aux-build:crateresolve1-1.rs
//@ aux-build:crateresolve1-2.rs
//@ aux-build:crateresolve1-3.rs

//@ normalize-stderr: "crateresolve1\..+/auxiliary/" -> "crateresolve1/auxiliary/"
//@ normalize-stderr: "\\\?\\" -> ""
//@ normalize-stderr: "(lib)?crateresolve1-([123])\.[a-z]+" -> "libcrateresolve1-$2.somelib"

// NOTE: This test is duplicated at `tests/ui/error-codes/E0464.rs` and `E0523.rs`.

extern crate crateresolve1;
//~^ ERROR multiple candidates for `rlib` dependency `crateresolve1` found

fn main() {}
