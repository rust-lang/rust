//! Regression test for <https://github.com/rust-lang/rust/issues/162393>.
//@ compile-flags: -Znext-solver=globally
//@ check-fail

#![feature(gca_macroless_args)]
#![feature(gca_const_items, gca_min_const_items)]
const C_INNER: (*const u8, u8) = (None::<u8>, None::<u8>);
//~^ ERROR mismatched types
//~| ERROR mismatched types

fn foo2(x: *const u8) {
    match (x, 1) {
        C_INNER => {} //~ ERROR could not evaluate constant pattern
        _ => {}
    }
}

fn main() {}
