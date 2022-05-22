// This tests that `default_method_body_is_const` methods can
// be called from a const context when used across crates.
//
// check-pass

#![feature(const_trait_impl)]

// aux-build: cross-crate.rs
extern crate cross_crate;

use cross_crate::*;

const _: () = {
    Const.func();
    Const.defaulted_func();
};

fn main() {}
