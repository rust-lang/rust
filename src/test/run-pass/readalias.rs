#![allow(dead_code)]




struct Point {x: isize, y: isize, z: isize}

fn f(p: Point) { assert_eq!(p.z, 12); }

pub fn main() { let x: Point = Point {x: 10, y: 11, z: 12}; f(x); }
