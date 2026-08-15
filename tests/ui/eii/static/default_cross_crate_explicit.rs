//@ aux-build: decl_with_default.rs
//@ aux-build: impl_default_override.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME(#125418): linking on Windows GNU targets is not yet supported.
//@ ignore-windows-gnu
// FIXME(#157649): static EII defaults currently fail to link on Apple targets.
//@ ignore-apple
// Tests that an explicit static EII implementation overrides a cross-crate default.

extern crate decl_with_default;
extern crate impl_default_override;

fn main() {
    println!("{}", decl_with_default::DECL1);
}
