//! Regression test for <https://github.com/rust-lang/rust/issues/134060> due to impl bug from
//! <https://github.com/rust-lang/rust/pull/131669>. This test should be adjusted in favor of more
//! comprehensive coverage when the changes are to be relanded, as this is a basic sanity check to
//! check that the fuzzed example from #134060 doesn't ICE.

//@ check-pass

#![crate_type = "lib"]

pub trait Foo {
    extern "C" fn foo_(&self, _: ()) -> i64 {
        //~^ WARN `extern` fn uses type `()`, which is not FFI-safe
        0
    }
}
