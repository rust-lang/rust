// issue: <https://github.com/rust-lang/rust/issues/14422>
// Test that we can call an inherently implemented method via aliasing from an extern crate.
//@ run-pass
#![allow(non_snake_case)]

//@ aux-build:extern-crate-alias-impl-method-aux.rs

extern crate extern_crate_alias_impl_method_aux as bug_lib;

use bug_lib::B;
use bug_lib::make;

pub fn main() {
    let mut an_A: B = make();
    an_A.foo();
}
