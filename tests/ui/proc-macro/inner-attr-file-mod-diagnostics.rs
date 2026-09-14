//@ proc-macro: test-macros.rs
//@ edition:2024

#![feature(custom_inner_attributes)]

extern crate test_macros;

#[path = "auxiliary/inner_attr_file_mod_diagnostics_child.rs"]
pub mod child;

fn main() {}

//~? ERROR cannot find function `g` in this scope
