//@ aux-build: decl_with_default.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests that a static EII default can be used from another crate.

extern crate decl_with_default;

fn main() {
    println!("{}", decl_with_default::DECL1);
}
