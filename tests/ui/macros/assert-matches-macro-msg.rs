//@ run-fail
//@ check-run-results:assertion `left matches right` failed: 1 + 1 definitely should be 3
//@ check-run-results:  left: 2
//@ check-run-results: right: 3
//@ ignore-emscripten no processes

#![feature(assert_matches)]

use std::assert_matches::assert_matches;

fn main() {
    assert_matches!(1 + 1, 3, "1 + 1 definitely should be 3");
}
