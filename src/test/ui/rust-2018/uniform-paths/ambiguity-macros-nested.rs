// edition:2018

#![feature(uniform_paths)]

// This test is similar to `ambiguity-macros.rs`, but nested in a module.

mod foo {
    pub use std::io;
    //~^ ERROR `std` import is ambiguous

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
