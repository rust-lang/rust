// Tests that :pat in macros in edition 2021 allows top-level or-patterns.

//@ run-pass
//@ edition:2021

macro_rules! accept_pat {
    ($p:pat) => {};
}

accept_pat!(p | q);

fn main() {}
