//! Regression test for <https://github.com/rust-lang/rust/issues/38875>.
//! This used to check `constant evaluation error` span doesn't point
//! on nothing when const is defined in another crate.
//@ aux-build:cross-crate-array-len.rs
//@ check-pass

extern crate cross_crate_array_len;

fn main() {
    let test_x = [0; cross_crate_array_len::FOO];
}
