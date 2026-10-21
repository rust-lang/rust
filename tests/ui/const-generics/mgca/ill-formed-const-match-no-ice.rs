//@ compile-flags: -Znext-solver=globally
//@ check-fail

// Regression test for <https://github.com/rust-lang/rust/issues/162389>.

#![allow(incomplete_features)]
#![feature(gca_macroless_args, gca_const_items, gca_min_const_items)]

const C_INNER: (*const u8, u8) = (C, 0);
//~^ ERROR: cannot find value `C` in this scope
fn foo2(x: *const u8) {
    match (x, 1) {
        C_INNER => {} //~ ERROR: could not evaluate constant pattern
    };
}

fn main() {}
