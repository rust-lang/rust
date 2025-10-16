//! Misused internal attributes should not be able to cause an ICE if
//! `#![feature(rustc_attr)]` is not even enabled.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/147761>.

//@ revisions: check build
//@ compile-flags: --crate-type=lib
//@ [check] check-fail
//@ [build] build-fail
//@ [build] dont-require-check-pass
//@ [build] failure-status: 1

fn array() {
    //~v ERROR use of an internal attribute
    #[rustc_layout_scalar_valid_range_start(1)]
    struct NonZero<T>([T; 4]);
    let nums = [1, 2, 3, 4];
    let _ = unsafe { NonZero(nums) };
}
