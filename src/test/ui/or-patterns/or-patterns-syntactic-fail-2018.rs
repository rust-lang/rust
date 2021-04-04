// Test that :pat doesn't accept top-level or-patterns in edition 2018.

// edition:2018

#![feature(or_patterns)]

fn main() {}

// Test the `pat` macro fragment parser:
macro_rules! accept_pat {
    ($p:pat) => {};
}

accept_pat!(p | q); //~ ERROR no rules expected the token `|`
accept_pat!(|p| q); //~ ERROR no rules expected the token `|`
