//@ no-prefer-dynamic
//@ aux-build: decl_with_default.rs
//@ aux-build: impl1.rs
//@ run-pass
//@ check-run-results
#![feature(eii)]

extern crate decl_with_default;
extern crate impl1;

fn main() {
    decl_with_default::decl1(10);
}
