//@ run-pass
//@ aux-build:const-fn-evaluated-cross-crate.rs
//! Regression test for https://github.com/rust-lang/rust/issues/36954
//! Cross crate const fn evaluation failed because the compiler's
//! internal argument lookup table used crate local ids that didn't
//! exist when evaluating a const fn from another crate.

extern crate const_fn_evaluated_cross_crate as lib;

fn main() {
    let _ = lib::FOO;
}
