// edition:2018

// This test is similar to `ambiguity.rs`, but with macros defining local items.

#![allow(non_camel_case_types)]

use std::io;
//~^ ERROR `std` is ambiguous

macro_rules! m {
    () => {
        mod std {
            pub struct io;
        }
    }
}
m!();

fn main() {}
