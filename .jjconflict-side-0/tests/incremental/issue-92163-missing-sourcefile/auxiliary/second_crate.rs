//@ compile-flags:--extern first_crate

// Note: adding `first_crate` to the extern prelude
// (instead of using `extern_crate`) appears to be necessary to
// trigger the original incremental compilation bug.
// I'm not entirely sure why this is the case

pub fn make_it() -> first_crate::Foo {
    panic!()
}
