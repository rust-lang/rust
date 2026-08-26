// Ensure that we parse `'r#lt` as three tokens pre Rust 2021.
// Moreover, make sure we emit the relevant migration lint.

//@ edition: 2015..2021
//@ check-pass

#![warn(rust_2021_prefixes_incompatible_syntax)]

macro_rules! ed2015 {
    ('r # lt) => {};
    ($lt:lifetime) => { compile_error!() };
}

ed2015!('r#lt);
//~^ WARNING prefix `'r` is reserved
//~| WARNING hard error in Rust 2021

fn main() {}
