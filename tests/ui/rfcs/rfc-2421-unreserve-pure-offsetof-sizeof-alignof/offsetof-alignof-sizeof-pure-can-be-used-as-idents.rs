//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test that removed keywords are allowed as identifiers.
fn main () {
    let offsetof = ();
    let alignof = ();
    let sizeof = ();
    let pure = ();
}

fn offsetof() {}
fn alignof() {}
fn sizeof() {}
fn pure() {}
