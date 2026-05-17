//@ aux-build: decl_with_default.rs
//@ aux-build: impl_default_override.rs
//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// FIXME: linking on windows (specifically mingw) not yet supported, see tracking issue #125418
//@ ignore-windows
// Tests that an explicit static EII implementation overrides a cross-crate default.

extern crate decl_with_default;
extern crate impl_default_override;

fn main() {
    println!("{}", decl_with_default::DECL1);
}
