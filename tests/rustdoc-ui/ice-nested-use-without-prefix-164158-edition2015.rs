//@ check-pass
//@ edition: 2015

// Regression test for <https://github.com/rust-lang/rust/issues/164158>.
// Nested glob use with no path prefix (edition 2015 form from review).

mod what {
    use {*};
}
