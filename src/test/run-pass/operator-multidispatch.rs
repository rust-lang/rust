// Test that we can overload the `+` operator for points so that two
// points can be added, and a point can be added to an integer.

use std::ops;

#[derive(Debug,PartialEq,Eq)]
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

impl ops::Add<isize> for Point {
    type Output = Point;

    fn add(self, other: isize) -> Point {
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
