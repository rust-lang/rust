// checks whether closures work in guard patterns

//@ check-pass
#![feature(guard_patterns)]

pub fn probe((_ if (|| true)()): u8) {}

fn main() {}
