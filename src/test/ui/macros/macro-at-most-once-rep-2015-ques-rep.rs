// Test behavior of `?` macro _kleene op_ under the 2015 edition. Namely, it doesn't exist.

// edition:2015

macro_rules! bar {
    ($(a)?) => {} //~ERROR expected `*` or `+`
}

macro_rules! baz {
    ($(a),?) => {} //~ERROR expected `*` or `+`
}

fn main() {}
