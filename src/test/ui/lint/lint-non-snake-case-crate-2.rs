// compile-flags: --crate-name NonSnakeCase
// error-pattern: crate `NonSnakeCase` should have a snake case name

#![deny(non_snake_case)]

fn main() {}
