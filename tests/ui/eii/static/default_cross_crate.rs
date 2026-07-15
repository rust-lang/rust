//@ aux-build: decl_with_default.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// FIXME(#157649): static EII defaults currently fail to link on Apple targets.
//@ ignore-apple
// Tests that a static EII default can be used from another crate.

extern crate decl_with_default;

fn main() {
    println!("{}", decl_with_default::DECL1);
}
