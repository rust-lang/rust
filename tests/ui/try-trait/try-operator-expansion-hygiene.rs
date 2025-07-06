//! Checks the `?` operator expansion.

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
