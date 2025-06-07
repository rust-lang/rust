//! Checks error handling for mismatched `--crate-name` and `#![crate_name]` values.

//@ compile-flags: --crate-name foo

#![crate_name = "bar"]
//~^ ERROR: `--crate-name` and `#[crate_name]` are required to match, but `foo` != `bar`

fn main() {}
