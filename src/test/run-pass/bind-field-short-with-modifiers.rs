// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    struct Foo { x: int, y: int }
    let mut f = Foo { x: 10, y: 0 };
    match f {
        Foo { ref mut x, .. } => *x = 11,
    }
    match f {
        Foo { ref x, ref y } => {
            assert_eq!(f.x, 11);
            assert_eq!(f.y, 0);
        }
    }
    match f {
        Foo { mut x, y: ref mut y } => {
            x = 12;
            *y = 1;
        }
    }
    assert_eq!(f.x, 11);
    assert_eq!(f.y, 1);
}
