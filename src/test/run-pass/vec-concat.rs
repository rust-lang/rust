// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let a: ~[int] = ~[1, 2, 3, 4, 5];
    let b: ~[int] = ~[6, 7, 8, 9, 0];
    let v: ~[int] = a + b;
    debug!(v[9]);
    assert!((v[0] == 1));
    assert!((v[7] == 8));
    assert!((v[9] == 0));
}
