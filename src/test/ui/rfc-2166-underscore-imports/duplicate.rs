// compile-pass
// aux-build:duplicate.rs

#![feature(underscore_imports)]

extern crate duplicate;

#[duplicate::duplicate]
use main as _; // OK

macro_rules! duplicate {
    ($item: item) => { $item $item }
}

duplicate!(use std as _;); // OK

fn main() {}
