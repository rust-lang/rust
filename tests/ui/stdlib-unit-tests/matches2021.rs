//@ run-pass
//@ edition:2021

// regression test for https://github.com/rust-lang/rust/pull/85678

use std::macros::assert_matches;

fn main() {
    assert!(matches!((), ()));
    assert_matches!((), ());
}
