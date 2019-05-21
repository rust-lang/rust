// aux-build:test-macros.rs

// FIXME: https://github.com/rust-lang/rust/issues/41430
// This is a temporary regression test for the ICE reported in #41211

#![feature(custom_inner_attributes)]

#![identity_attr]
//~^ ERROR attribute `identity_attr` is currently unknown to the compiler
//~| ERROR inconsistent resolution for a macro: first custom attribute, then attribute macro
extern crate test_macros;
use test_macros::identity_attr;

fn main() {}
