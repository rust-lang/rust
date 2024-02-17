//@ check-pass
//@ aux-build:dollar-crate-nested-encoding.rs

extern crate dollar_crate_nested_encoding;

type A = dollar_crate_nested_encoding::exported!();

fn main() {}
