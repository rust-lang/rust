//@ aux-build: decl_with_default.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// Tests EIIs with default implementations.
// When there's no explicit declaration, the default should be called from the declaring crate.

extern crate decl_with_default;

fn main() {
    decl_with_default::decl1(10);
}
