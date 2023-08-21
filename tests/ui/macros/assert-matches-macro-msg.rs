// run-fail
// error-pattern:assertion `left matches right` failed: 1 + 1 definitely should be 3
// error-pattern:  left: 2
// error-pattern: right: 3
// ignore-emscripten no processes

#![feature(assert_matches)]

use std::assert_matches::assert_matches;

fn main() {
    assert_matches!(1 + 1, 3, "1 + 1 definitely should be 3");
}
