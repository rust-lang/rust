// Test that enabling an unstable feature disables warnings

//@ aux-build:stability-cfg2.rs

#![feature(unstable_test_feature)]
#![deny(non_snake_case)] // To trigger a hard error

// Shouldn't generate a warning about unstable features
extern crate stability_cfg2;

pub fn BOGUS() { } //~ ERROR

pub fn main() { }
