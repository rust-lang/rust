//@ aux-build: decl_with_default.rs
//@ aux-build: impl1.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests EIIs with default implementations.
// When an explicit implementation is given in one dependency, and the declaration is in another,
// the explicit implementation is preferred.

extern crate decl_with_default;
extern crate impl1;

fn main() {
    decl_with_default::decl1(10);
}
