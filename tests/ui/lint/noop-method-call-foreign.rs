//! Regression test for <https://github.com/rust-lang/rust/issues/134471>.
//!
//! The derive suggestion must not appear for a type from another crate, because the user can't
//! edit that type.

//@ check-pass
//@ aux-build:noop-method-call-foreign.rs

extern crate noop_method_call_foreign;

use noop_method_call_foreign::Foreign;

pub fn foo(v: &Foreign) {
    let _v = v.clone();
    //~^ WARN call to `.clone()` on a reference in this situation does nothing
}

fn main() {}
