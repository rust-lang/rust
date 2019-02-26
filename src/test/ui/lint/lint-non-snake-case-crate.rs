#![crate_name = "NonSnakeCase"]
//~^ ERROR crate `NonSnakeCase` should have a snake case name
#![deny(non_snake_case)]

fn main() {}
