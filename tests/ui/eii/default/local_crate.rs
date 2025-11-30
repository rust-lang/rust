//@ run-pass
//@ check-run-results
//@ ignore-backends: gcc
// Tests EIIs with default implementations.
// In the same crate, when there's no explicit declaration, the default should be called.
#![feature(eii)]

#[eii(eii1)]
pub fn decl1(x: u64) {
    println!("default {x}");
}

fn main() {
    decl1(4);
}
