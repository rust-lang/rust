// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `expr?` expands to:
//
// match expr {
//     Ok(val) => val,
//     Err(err) => return Err(From::from(err)),
// }
//
// This test verifies that the expansion is hygienic, i.e. it's not affected by other `val` and
// `err` bindings that may be in scope.

use std::num::ParseIntError;

fn main() {
    assert_eq!(parse(), Ok(1));
}

fn parse() -> Result<i32, ParseIntError> {
    const val: char = 'a';
    const err: char = 'b';

    Ok("1".parse::<i32>()?)
}
