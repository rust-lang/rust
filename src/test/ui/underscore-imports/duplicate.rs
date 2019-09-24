// build-pass (FIXME(62277): could be check-pass?)
// aux-build:duplicate.rs

extern crate duplicate;

#[duplicate::duplicate]
use main as _; // OK

macro_rules! duplicate {
    ($item: item) => { $item $item }
}

duplicate!(use std as _;); // OK

fn main() {}
