// Tests that :pat in macros in edition 2021 allows top-level or-patterns.

// run-pass
// ignore-test
// edition:2021
// FIXME(mark-i-m): unignore when 2021 machinery is in place.

macro_rules! accept_pat {
    ($p:pat) => {};
}

accept_pat!(p | q);

fn main() {}
