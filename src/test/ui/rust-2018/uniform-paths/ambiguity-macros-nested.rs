// edition:2018

// This test is similar to `ambiguity-macros.rs`, but nested in a module.

#![allow(non_camel_case_types)]

mod foo {
    pub use std::io;
    //~^ ERROR `std` is ambiguous

    macro_rules! m {
        () => {
            mod std {
                pub struct io;
            }
        }
    }
    m!();
}

fn main() {}
