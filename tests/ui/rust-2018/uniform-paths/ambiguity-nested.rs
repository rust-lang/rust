//@ check-pass
//@ edition:2018

// This test is similar to `ambiguity.rs`, but nested in a module.

#![allow(non_camel_case_types)]

mod foo {
    pub use std::io; // OK

    mod std {
        pub struct io;
    }
}

fn main() {}
