//! Regression test for <https://github.com/rust-lang/rust/issues/18501>.
//!
//! Test that we don't ICE when inlining a function from another
//! crate that uses a trait method as a value due to incorrectly
//! translating the def ID of the trait during AST decoding.

//@ run-pass

//@ aux-build:inline-fn-with-trait-method-as-value.rs

extern crate inline_fn_with_trait_method_as_value as issue;

fn main() {
    issue::pass_method();
}
