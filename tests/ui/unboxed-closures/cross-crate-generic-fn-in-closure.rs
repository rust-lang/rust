//! Regression test for <https://github.com/rust-lang/rust/issues/18711>.
//! Test that we don't panic on a RefCell borrow conflict in certain
//! code paths involving unboxed closures.

//@ run-pass

//@ aux-build:cross-crate-generic-fn-in-closure.rs
extern crate cross_crate_generic_fn_in_closure as issue;

fn main() {
    (|| issue::inner(()))();
}
