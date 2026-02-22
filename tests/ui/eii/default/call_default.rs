//@ no-prefer-dynamic
//@ aux-build: decl_with_default.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (speciifcally mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests EIIs with default implementations.
// When there's no explicit declaration, the default should be called from the declaring crate.
#![feature(extern_item_impls)]

extern crate decl_with_default;

fn main() {
    decl_with_default::decl1(10);
}
