//@ no-prefer-dynamic
//@ aux-build: decl_with_default.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// Tests EIIs with default implementations.
// When there's no explicit declaration, the default should be called from the declaring crate.
#![feature(eii)]

extern crate decl_with_default;

fn main() {
    decl_with_default::decl1(10);
}
