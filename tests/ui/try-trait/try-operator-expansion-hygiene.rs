//! This test verifies that the `?` operator expansion is hygienic,
//! i.e., it's not affected by other `val` and `err` bindings that may be in scope.
//!
//! Note: Prior to the Try trait stabilization, `expr?` expanded to a match
//! with `val` and `err` bindings. The current implementation uses Try::branch()
//! but this test remains relevant for hygiene verification.

//@ run-pass

#![allow(non_upper_case_globals)]
#![allow(dead_code)]

use std::num::ParseIntError;

fn main() {
    assert_eq!(parse(), Ok(1));
}

fn parse() -> Result<i32, ParseIntError> {
    const val: char = 'a';
    const err: char = 'b';

    Ok("1".parse::<i32>()?)
}
