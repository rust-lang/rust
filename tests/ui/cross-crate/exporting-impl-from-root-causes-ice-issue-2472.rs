//@ run-pass
//@ aux-build:exporting-impl-from-root-causes-ice-issue-2472-b.rs


extern crate exporting_impl_from_root_causes_ice_issue_2472_b as lib;

use lib::{S, T};

pub fn main() {
    let s = S(());
    s.foo();
    s.bar();
}

// https://github.com/rust-lang/rust/issues/2472
