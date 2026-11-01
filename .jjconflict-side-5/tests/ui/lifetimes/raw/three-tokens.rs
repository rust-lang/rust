// Ensure that we parse `'r#lt` as three tokens pre Rust 2021.
// Moreover, make sure we emit the relevant migration lint.

//@ edition: 2015..2021
//@ check-pass

#![warn(rust_2021_prefixes_incompatible_syntax)]

macro_rules! check {
    ('r # lt) => {};
    ($lt:lifetime) => { compile_error!() };
}

check!('r#lt);
//~^ WARNING parsed as a prefix in Rust 2021 and onward
//~| WARNING this changes meaning in Rust 2021

fn main() {}
