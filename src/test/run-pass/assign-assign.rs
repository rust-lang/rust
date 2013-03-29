// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 483 - Assignment expressions result in nil
fn test_assign() {
    let mut x: int;
    let mut y: () = x = 10;
    assert!((x == 10));
    let mut z = x = 11;
    assert!((x == 11));
    z = x = 12;
    assert!((x == 12));
}

fn test_assign_op() {
    let mut x: int = 0;
    let mut y: () = x += 10;
    assert!((x == 10));
    let mut z = x += 11;
    assert!((x == 21));
    z = x += 12;
    assert!((x == 33));
}

pub fn main() { test_assign(); test_assign_op(); }
