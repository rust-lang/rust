// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::vec;

pub fn main() {
    fn f(i: uint) -> uint { i }
    let v = ~[-1f, 0f, 1f, 2f, 3f];
    let z = do vec::foldl(f, v) |x, _y| { x } (22u);
    assert_eq!(z, 22u);
}
