//@ run-pass
//@ aux-build:exporting-impl-from-root-causes-ice-2472-b.rs
//@ ignore-ios FIXME(madsmtm): For some reason the necessary dylib isn't copied to the remote?
//@ ignore-tvos FIXME(madsmtm): For some reason the necessary dylib isn't copied to the remote?
//@ ignore-watchos FIXME(madsmtm): For some reason the necessary dylib isn't copied to the remote?
//@ ignore-visionos FIXME(madsmtm): For some reason the necessary dylib isn't copied to the remote?

extern crate exporting_impl_from_root_causes_ice_2472_b as lib;

use lib::{S, T};

pub fn main() {
    let s = S(());
    s.foo();
    s.bar();
}

// https://github.com/rust-lang/rust/issues/2472
