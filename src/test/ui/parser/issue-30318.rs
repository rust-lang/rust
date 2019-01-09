fn foo() { }

//! Misplaced comment...
//~^ ERROR expected outer doc comment
//~| NOTE inner doc comments like this (starting with `//!` or `/*!`) can only appear before items

fn main() { }
