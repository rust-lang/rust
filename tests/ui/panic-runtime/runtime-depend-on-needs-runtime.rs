//@ dont-check-compiler-stderr
//@ aux-build:needs-panic-runtime.rs
//@ aux-build:depends.rs

extern crate depends;

fn main() {}

//~? ERROR the crate `depends` cannot depend on a crate that needs a panic runtime, but it depends on `needs_panic_runtime`
