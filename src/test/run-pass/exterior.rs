#![allow(dead_code)]


use std::cell::Cell;

#[derive(Copy, Clone)]
struct Point {x: isize, y: isize, z: isize}

fn f(p: &Cell<Point>) {
    assert_eq!(p.get().z, 12);
    p.set(Point {x: 10, y: 11, z: 13});
    assert_eq!(p.get().z, 13);
}

pub fn main() {
    let a: Point = Point {x: 10, y: 11, z: 12};
    let b: &Cell<Point> = &Cell::new(a);
    assert_eq!(b.get().z, 12);
    f(b);
    assert_eq!(a.z, 12);
    assert_eq!(b.get().z, 13);
}
