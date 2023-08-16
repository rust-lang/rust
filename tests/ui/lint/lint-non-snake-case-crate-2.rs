//@compile-flags: --crate-name NonSnakeCase
//@error-in-other-file: crate `NonSnakeCase` should have a snake case name

#![deny(non_snake_case)]

fn main() {}
