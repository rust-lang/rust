//! Checks proper validation of the `#![crate_name]` attribute.

//@ run-pass
//@ compile-flags:--crate-name crate_name_attr_used -F unused-attributes


#![crate_name = "crate_name_attr_used"]

fn main() {}
