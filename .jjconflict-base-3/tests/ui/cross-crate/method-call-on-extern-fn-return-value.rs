//@ edition:2018
//@ aux-build:method-call-on-extern-fn-return-value.rs
//@ check-pass
//! Regression test for https://github.com/rust-lang/rust/issues/51798
//! Calling a method on a value returned from an external crate function
//! caused the privacy checker to panic when looking up the method's
//! type dependent definition. Requires 2018 edition.

extern crate method_call_on_extern_fn_return_value as issue_51798;

mod server {
    fn f() {
        let mut v = issue_51798::vec();
        v.clear();
    }
}

fn main() {}
