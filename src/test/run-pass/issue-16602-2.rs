// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A {
    pub x: u32,
    pub y: u32,
}

fn main() {
    let mut a = A { x: 1, y: 1 };
    a = A { x: a.y * 2, y: a.x * 2 };
    assert_eq!(a.x, 2);
    assert_eq!(a.y, 2);
}
