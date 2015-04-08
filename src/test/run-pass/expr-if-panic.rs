// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn test_if_panic() {
    let x = if false { panic!() } else { 10 };
    assert!((x == 10));
}

fn test_else_panic() {
    let x = if true { 10 } else { panic!() };
    assert_eq!(x, 10);
}

fn test_elseif_panic() {
    let x = if false { 0 } else if false { panic!() } else { 10 };
    assert_eq!(x, 10);
}

pub fn main() { test_if_panic(); test_else_panic(); test_elseif_panic(); }
