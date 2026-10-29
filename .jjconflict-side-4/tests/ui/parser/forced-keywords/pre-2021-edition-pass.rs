//@ edition: 2015..2021
//@ check-pass

#![warn(rust_2021_prefixes_incompatible_syntax)]

macro_rules! ensure {
    ($tag:ident # $name:ident) => {};
}

ensure! { k#fn }
//~^ WARNING parsed as a forced keyword in Rust 2021 and onward
//~| WARNING this changes meaning in Rust 2021

fn main() {}
