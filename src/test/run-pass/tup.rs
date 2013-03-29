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

type point = (int, int);

fn f(p: point, x: int, y: int) {
    let (a, b) = p;
    assert!((a == x));
    assert!((b == y));
}

pub fn main() {
    let p: point = (10, 20);
    let (a, b) = p;
    assert!((a == 10));
    assert!((b == 20));
    let p2: point = p;
    f(p, 10, 20);
    f(p2, 10, 20);
}
