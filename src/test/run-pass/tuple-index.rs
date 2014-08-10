// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(tuple_indexing)]

struct Point(int, int);

fn main() {
    let mut x = Point(3, 2);
    assert_eq!(x.0, 3);
    assert_eq!(x.1, 2);
    x.0 += 5;
    assert_eq!(x.0, 8);
    {
        let ry = &mut x.1;
        *ry -= 2;
        x.0 += 3;
        assert_eq!(x.0, 11);
    }
    assert_eq!(x.1, 0);

    let mut x = (3i, 2i);
    assert_eq!(x.0, 3);
    assert_eq!(x.1, 2);
    x.0 += 5;
    assert_eq!(x.0, 8);
    {
        let ry = &mut x.1;
        *ry -= 2;
        x.0 += 3;
        assert_eq!(x.0, 11);
    }
    assert_eq!(x.1, 0);

}
