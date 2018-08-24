// edition:2018

#![feature(uniform_paths)]

// This test is similar to `ambiguity.rs`, but with macros defining local items.

use std::io;
//~^ ERROR `std` import is ambiguous

macro_rules! m {
    () => {
        mod std {
            pub struct io;
        }
    }
}
m!();

fn main() {}
