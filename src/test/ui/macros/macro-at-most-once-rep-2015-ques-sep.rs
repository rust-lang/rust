// Test behavior of `?` macro _separator_ under the 2015 edition. Namely, `?` can be used as a
// separator, but you get a migration warning for the edition.

// edition:2015
// compile-pass

#![warn(rust_2018_compatibility)]

macro_rules! bar {
    ($(a)?*) => {} //~WARN using `?` as a separator
    //~^WARN this was previously accepted
}

macro_rules! baz {
    ($(a)?+) => {} //~WARN using `?` as a separator
    //~^WARN this was previously accepted
}

fn main() {
    bar!();
    bar!(a);
    bar!(a?a);
    bar!(a?a?a?a?a);

    baz!(a);
    baz!(a?a);
    baz!(a?a?a?a?a);
}
