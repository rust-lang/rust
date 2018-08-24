// edition:2018

#![feature(uniform_paths)]

// This test is similar to `ambiguity.rs`, but nested in a module.

mod foo {
    pub use std::io;
    //~^ ERROR `std` import is ambiguous

    mod std {
        pub struct io;
    }
}

fn main() {}
