// aux-build:lint-for-crate.rs
// ignore-stage1
// compile-flags: -D crate-not-okay

#![feature(plugin)] //~ ERROR crate is not marked with #![crate_okay]
#![plugin(lint_for_crate)]

pub fn main() { }
