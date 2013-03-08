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
    fail_unless!((r.x == x));
    fail_unless!((r.y == y));
    fail_unless!((r.w == w));
    fail_unless!((r.h == h));
}

pub fn main() {
    let r: Rect = Rect {x: 10, y: 20, w: 100, h: 200};
    fail_unless!((r.x == 10));
    fail_unless!((r.y == 20));
    fail_unless!((r.w == 100));
    fail_unless!((r.h == 200));
    let r2: Rect = r;
    let x: int = r2.x;
    fail_unless!((x == 10));
    f(r, 10, 20, 100, 200);
    f(r2, 10, 20, 100, 200);
}
