//@ run-fail
//@ error-pattern:beep boop
//@ needs-subprocess

#![allow(unused_variables)]

struct Point {
    x: isize,
    y: isize,
}

fn main() {
    let origin = Point { x: 0, y: 0 };
    let f: Point = Point { x: (panic!("beep boop")), ..origin };
}
