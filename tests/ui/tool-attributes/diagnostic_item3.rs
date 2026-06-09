//@ check-pass
#![feature(rustc_attrs)]

#[rustc_diagnostic_item = "foomp"]
struct Foomp;

fn main() {}
