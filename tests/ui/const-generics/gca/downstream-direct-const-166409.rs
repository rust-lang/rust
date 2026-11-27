//! Regression test for <https://github.com/rust-lang/rust/issues/166409>.
//@ aux-build: gca-direct-const-upstream-166409.rs
//@ run-pass

extern crate gca_direct_const_upstream_166409 as up;

use up::Tr;

const N: usize = up::N;

fn main() {
    assert_eq!(N, 1);
    assert_eq!(<up::S as Tr>::C, 3);
    assert_eq!(<up::S as Tr>::D, 5);
}
