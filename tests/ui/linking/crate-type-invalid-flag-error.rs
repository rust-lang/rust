// Test for #70183 that --crate-type flag display valid value.

//@ compile-flags: --crate-type dynlib

fn main() {}

//~? ERROR unknown crate type: `dynlib`
