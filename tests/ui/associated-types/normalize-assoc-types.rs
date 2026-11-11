//! Regression test for <https://github.com/rust-lang/rust/issues/22629>.
//! Test transitive analysis for associated types. Collected types
//! should be normalized and new obligations generated.

//@ run-pass
#![allow(unused_imports)]

use std::borrow::{ToOwned, Cow};

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(Cow::Borrowed("foo"));
}
