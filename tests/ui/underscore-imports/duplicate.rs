//@ check-pass
//@ proc-macro: duplicate.rs

extern crate duplicate;

#[duplicate::duplicate]
use main as _; // OK

macro_rules! duplicate {
    ($item: item) => { $item $item }
}

duplicate!(use std as _;); // OK

fn main() {}
