//@ run-pass
#![allow(dead_code)]

struct Point {x: isize, y: isize}

fn x_coord(p: &Point) -> &isize {
    return &p.x;
}

pub fn main() {
    let p: Box<_> = Box::new(Point {x: 3, y: 4});
    let xc = x_coord(&*p);
    assert_eq!(*xc, 3);
}
