// run-fail
// check-run-results
// ignore-emscripten no processes

#![feature(assert_matches)]

use std::assert_matches::assert_matches;

fn main() {
    assert_matches!(1 + 1, 3, "1 + 1 definitely should be 3");
}
