// Test behavior of `?` macro _kleene op_ under the 2015 edition. Namely, it doesn't exist, even
// with the feature flag.

// gate-test-macro_at_most_once_rep
// edition:2015

#![feature(macro_at_most_once_rep)]

macro_rules! bar {
    ($(a)?) => {} //~ERROR expected `*` or `+`
}

macro_rules! baz {
    ($(a),?) => {} //~ERROR expected `*` or `+`
}

fn main() {}

