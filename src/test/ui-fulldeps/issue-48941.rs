// This is a regression test against an ICE that used to occur
// on malformed attributes for a custom MultiModifier.

// aux-build:macro_crate_test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(macro_crate_test)]

#[noop_attribute("hi", rank = a)] //~ ERROR expected unsuffixed literal or identifier, found a
fn knight() { }

#[noop_attribute("/user", data= = "<user")] //~ ERROR literal or identifier
fn nite() { }

fn main() {}
