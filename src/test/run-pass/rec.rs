// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-
struct Rect {x: int, y: int, w: int, h: int}

fn f(r: Rect, x: int, y: int, w: int, h: int) {
    assert_eq!(r.x, x);
    assert_eq!(r.y, y);
    assert_eq!(r.w, w);
    assert_eq!(r.h, h);
}

pub fn main() {
    let r: Rect = Rect {x: 10, y: 20, w: 100, h: 200};
    assert_eq!(r.x, 10);
    assert_eq!(r.y, 20);
    assert_eq!(r.w, 100);
    assert_eq!(r.h, 200);
    let r2: Rect = r;
    let x: int = r2.x;
    assert_eq!(x, 10);
    f(r, 10, 20, 100, 200);
    f(r2, 10, 20, 100, 200);
}
