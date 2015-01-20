// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can overload the `+` operator for points so that two
// points can be added, and a point can be added to an integer.

use std::ops;

#[derive(Show,PartialEq,Eq)]
struct Point {
    x: int,
    y: int
}

impl ops::Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {x: self.x + other.x, y: self.y + other.y}
    }
}

impl ops::Add<int> for Point {
    type Output = Point;

    fn add(self, other: int) -> Point {
        Point {x: self.x + other,
               y: self.y + other}
    }
}

pub fn main() {
    let mut p = Point {x: 10, y: 20};
    p = p + Point {x: 101, y: 102};
    assert_eq!(p, Point {x: 111, y: 122});
    p = p + 1;
    assert_eq!(p, Point {x: 112, y: 123});
}
