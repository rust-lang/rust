// run-fail
//@error-in-other-file:panicked at 'assertion failed: `(left matches right)`
//@error-in-other-file: left: `2`
//@error-in-other-file:right: `3`: 1 + 1 definitely should be 3'
//@ignore-target-emscripten no processes

#![feature(assert_matches)]

use std::assert_matches::assert_matches;

fn main() {
    assert_matches!(1 + 1, 3, "1 + 1 definitely should be 3");
}
