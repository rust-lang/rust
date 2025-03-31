//@ no-prefer-dynamic
//@ aux-build: decl_with_default.rs
//@ run-pass
//@ check-run-results
#![feature(eii)]

extern crate decl_with_default;

fn main() {
    decl_with_default::decl1(10);
}
