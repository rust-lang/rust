//@ run-pass
#![allow(dead_code)]


struct Point {
    x: isize,
    y: isize
}

struct Character {
    pos: Box<Point>,
}

fn get_x(x: &Character) -> &isize {
    // interesting case because the scope of this
    // borrow of the unique pointer is in fact
    // larger than the fn itself
    return &x.pos.x;
}

pub fn main() {
}
