// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.





// Tests for standalone blocks as expressions
fn test_basic() { let rs: bool = { true }; assert!((rs)); }

struct RS { v1: int, v2: int }

fn test_rec() { let rs = { RS {v1: 10, v2: 20} }; assert!((rs.v2 == 20)); }

fn test_filled_with_stuff() {
    let rs = { let mut a = 0i; while a < 10 { a += 1; } a };
    assert_eq!(rs, 10);
}

pub fn main() { test_basic(); test_rec(); test_filled_with_stuff(); }
