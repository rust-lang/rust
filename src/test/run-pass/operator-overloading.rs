// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

use std::cmp;
use std::ops;

struct Point {
    x: int,
    y: int
}

impl ops::Add<Point,Point> for Point {
    fn add(&self, other: &Point) -> Point {
        Point {x: self.x + (*other).x, y: self.y + (*other).y}
    }
}

impl ops::Sub<Point,Point> for Point {
    fn sub(&self, other: &Point) -> Point {
        Point {x: self.x - (*other).x, y: self.y - (*other).y}
    }
}

impl ops::Neg<Point> for Point {
    fn neg(&self) -> Point {
        Point {x: -self.x, y: -self.y}
    }
}

impl ops::Not<Point> for Point {
    fn not(&self) -> Point {
        Point {x: !self.x, y: !self.y }
    }
}

impl ops::Index<bool,int> for Point {
    fn index(&self, x: &bool) -> int {
        if *x { self.x } else { self.y }
    }
}

impl cmp::Eq for Point {
    fn eq(&self, other: &Point) -> bool {
        (*self).x == (*other).x && (*self).y == (*other).y
    }
    fn ne(&self, other: &Point) -> bool { !(*self).eq(other) }
}

pub fn main() {
    let mut p = Point {x: 10, y: 20};
    p += Point {x: 101, y: 102};
    p = p - Point {x: 100, y: 100};
    assert_eq!(p + Point {x: 5, y: 5}, Point {x: 16, y: 27});
    assert_eq!(-p, Point {x: -11, y: -22});
    assert_eq!(p[true], 11);
    assert_eq!(p[false], 22);

    let q = !p;
    assert_eq!(q.x, !(p.x));
    assert_eq!(q.y, !(p.y));

    // Issue #1733
    let result: ~fn(int) = |_|();
    result(p[true]);
}
