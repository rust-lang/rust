// run-pass
#![allow(unused_imports)]
// Test transitive analysis for associated types. Collected types
// should be normalized and new obligations generated.

// pretty-expanded FIXME #23616

use std::borrow::{ToOwned, Cow};

fn assert_send<T: Send>(_: T) {}

fn main() {
    assert_send(Cow::Borrowed("foo"));
}
