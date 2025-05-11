//@ check-pass
//@ edition:2021
//@ aux-build:non_local_macro.rs
//
// To suggest any Cargo specific help/note rustc wants
// the `CARGO_CRATE_NAME` env to be set, so we set it
//@ rustc-env:CARGO_CRATE_NAME=non_local_def
//
// and since we specifically want to check the presence
// of the `cargo update` suggestion we assert it here.
//@ dont-require-annotations: NOTE

extern crate non_local_macro;

struct LocalStruct;

non_local_macro::non_local_impl!(LocalStruct);
//~^ WARN non-local `impl` definition
//~| NOTE `cargo update -p non_local_macro`

fn main() {}
