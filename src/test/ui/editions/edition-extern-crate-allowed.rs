// aux-build:edition-extern-crate-allowed.rs
// edition:2015
// build-pass (FIXME(62277): could be check-pass?)

#![warn(rust_2018_idioms)]

extern crate edition_extern_crate_allowed;
//~^ WARNING unused extern crate

fn main() {}
