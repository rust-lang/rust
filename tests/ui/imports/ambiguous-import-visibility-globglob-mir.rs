// Regression test for the 1.96 -> 1.97 stable-to-stable regression: an item exported
// only through a public glob, and also glob-imported with restricted visibility through
// a private facade, lost its exported effective visibility. The defining crate then
// skipped encoding its optimized MIR (and warned dead_code) while name resolution still
// exported the item and it remained `cross_crate_inlinable`, so downstream crates failed
// with "missing optimized MIR". The dead_code half is checked by `#![deny(dead_code)]`
// in the auxiliary crate itself.

//@ build-pass
//@ aux-build:ambiguous-import-visibility-globglob-mir.rs

extern crate ambiguous_import_visibility_globglob_mir as dep;

pub fn call_f() -> u32 {
    dep::f()
}

fn main() {
    call_f();
}
