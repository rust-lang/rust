#![allow(unused_variables)]
use std::cmp;
use std::ops;

#[derive(Copy, Clone, Debug)]
struct Point {
    x: isize,
    y: isize
}

impl ops::Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {x: self.x + other.x, y: self.y + other.y}
    }
}

impl ops::Sub for Point {
    type Output = Point;

    fn sub(self, other: Point) -> Point {
        Point {x: self.x - other.x, y: self.y - other.y}
    }
}

impl ops::Neg for Point {
    type Output = Point;

    fn neg(self) -> Point {
        Point {x: -self.x, y: -self.y}
    }
}

impl ops::Not for Point {
    type Output = Point;

    fn not(self) -> Point {
        Point {x: !self.x, y: !self.y }
    }
}

impl ops::Index<bool> for Point {
    type Output = isize;

    fn index(&self, x: bool) -> &isize {
        if x {
            &self.x
        } else {
            &self.y
        }
    }
}

impl cmp::PartialEq for Point {
    fn eq(&self, other: &Point) -> bool {
        (*self).x == (*other).x && (*self).y == (*other).y
    }
    fn ne(&self, other: &Point) -> bool { !(*self).eq(other) }
}

pub fn main() {
    let mut p = Point {x: 10, y: 20};
    p = p + Point {x: 101, y: 102};
    p = p - Point {x: 100, y: 100};
    assert_eq!(p + Point {x: 5, y: 5}, Point {x: 16, y: 27});
    assert_eq!(-p, Point {x: -11, y: -22});
    assert_eq!(p[true], 11);
    assert_eq!(p[false], 22);

    let q = !p;
    assert_eq!(q.x, !(p.x));
    assert_eq!(q.y, !(p.y));

    // Issue #1733
    result(p[true]);
}

fn result(i: isize) { }
