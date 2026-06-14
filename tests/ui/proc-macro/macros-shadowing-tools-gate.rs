//@ proc-macro: rustfmt.rs
#![feature(prelude_import)]
#![rustfmt::skip]
//~^ ERROR: inner macro attributes are unstable

fn main() {}
