// compile-flags: --crate-name NonSnakeCase
// error-pattern: crate `NonSnakeCase` should have a snake case name such as `non_snake_case`

#![deny(non_snake_case)]

fn main() {}
