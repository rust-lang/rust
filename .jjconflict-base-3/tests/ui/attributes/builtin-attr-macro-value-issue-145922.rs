//@ check-pass
#![allow(unused_attributes, unused_macros)]

// Regression test for #145922.
// This used to create a delayed ICE while parsing builtin attributes.
#[crate_type = concat!("my", "crate")]
macro_rules! foo {
    () => {
        32
    };
}

#[recursion_limit = concat!("1", "2")]
macro_rules! baz {
    () => {
        32
    };
}

#[type_length_limit = concat!("1", "2")]
macro_rules! qux {
    () => {
        32
    };
}

#[windows_subsystem = concat!("cons", "ole")]
macro_rules! bar {
    () => {
        32
    };
}

fn main() {}
